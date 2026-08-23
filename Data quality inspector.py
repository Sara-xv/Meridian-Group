import pandas as pd
import numpy as np


class DataQualityInspector:
    """
    Data quality inspector for a single DataFrame.
    Does not modify the original data; everything is reporting only.
    """

    def __init__(self, df: pd.DataFrame, table_name: str = "unnamed_table",
                 expected_missing: dict | None = None):
        """
        Parameters
        ----------
        df : pd.DataFrame
            The table to inspect.
        table_name : str
            Table name, for report readability.
        expected_missing : dict, optional
            Mapping of column -> pandas condition (as string or callable) that specifies
            when missing values in that column are "expected" rather than a data issue.
            Example: {"assessment_score": lambda row: row["completion_status"] != "Completed"}
            If this condition is True for a row, that missing value is flagged as
            "expected", not as a "problem".
        """
        self.df = df
        self.table_name = table_name
        self.expected_missing = expected_missing or {}

    # ------------------------------------------------------------------
    # 1) MISSING VALUES
    # ------------------------------------------------------------------
    def check_missing(self) -> pd.DataFrame:
        """
        Report missing values per column, distinguishing between:
          - "expected" missing (according to rules defined in expected_missing)
          - "unexplained / needs review" missing (the rest)
        """
        rows = []
        n = len(self.df)
        for col in self.df.columns:
            null_mask = self.df[col].isna()
            n_missing = int(null_mask.sum())
            if n_missing == 0:
                continue

            n_expected = 0
            if col in self.expected_missing:
                rule = self.expected_missing[col]
                try:
                    expected_mask = self.df.apply(rule, axis=1) & null_mask
                    n_expected = int(expected_mask.sum())
                except Exception:
                    n_expected = 0

            n_unexplained = n_missing - n_expected
            rows.append({
                "column": col,
                "missing_count": n_missing,
                "missing_pct": round(n_missing / n * 100, 2),
                "explained_by_rule": n_expected,
                "unexplained_missing": n_unexplained,
                "needs_review": n_unexplained > 0,
            })

        empty_schema = {
            "column": pd.Series(dtype="object"),
            "missing_count": pd.Series(dtype="int64"),
            "missing_pct": pd.Series(dtype="float64"),
            "explained_by_rule": pd.Series(dtype="int64"),
            "unexplained_missing": pd.Series(dtype="int64"),
            "needs_review": pd.Series(dtype="bool"),
        }
        if not rows:
            return pd.DataFrame(empty_schema)

        result = pd.DataFrame(rows).sort_values("missing_count", ascending=False)
        return result.reset_index(drop=True)

    # ------------------------------------------------------------------
    # 2) DUPLICATES
    # ------------------------------------------------------------------
    def check_duplicates(self, subset: list | None = None) -> pd.DataFrame:
        """
        Returns duplicate rows (the rows themselves, not just counts) so you can
        see exactly which records are duplicates.

        subset: If None, checks entire row for duplication.
                If a list of columns is provided (e.g., ["employee_id"]), duplicates
                are checked based on that key (more important, since exact duplicate
                rows are rare; duplicate keys are more significant).
        """
        dup_mask = self.df.duplicated(subset=subset, keep=False)
        dup_rows = self.df[dup_mask].copy()
        if not dup_rows.empty:
            sort_cols = subset if subset else list(self.df.columns)
            dup_rows = dup_rows.sort_values(sort_cols)
        return dup_rows

    # ------------------------------------------------------------------
    # 3) OUTLIERS
    # ------------------------------------------------------------------
    def check_outliers(self, columns: list, method: str = "iqr",
                        iqr_multiplier: float = 1.5, z_threshold: float = 3.0) -> pd.DataFrame:
        """
        Flags numeric outliers in specified columns (does not remove them).

        method: "iqr" (default, more robust for skewed distributions) or "zscore".

        Output: A DataFrame with original columns + a boolean column for each
        inspected variable named '{col}_is_outlier', only including rows that
        have at least one outlier in any of the columns.
        """
        flags = pd.DataFrame(index=self.df.index)
        for col in columns:
            series = pd.to_numeric(self.df[col], errors="coerce")
            if method == "iqr":
                q1, q3 = series.quantile(0.25), series.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - iqr_multiplier * iqr
                upper = q3 + iqr_multiplier * iqr
                flags[f"{col}_is_outlier"] = (series < lower) | (series > upper)
            elif method == "zscore":
                mean, std = series.mean(), series.std()
                z = (series - mean) / std if std > 0 else 0
                flags[f"{col}_is_outlier"] = z.abs() > z_threshold
            else:
                raise ValueError("method must be 'iqr' or 'zscore'")

        any_outlier = flags.any(axis=1)
        result = self.df[any_outlier].copy()
        for c in flags.columns:
            result[c] = flags.loc[any_outlier, c]
        return result

    # ------------------------------------------------------------------
    # 4) INVALID / LOGICALLY INCONSISTENT VALUES
    # ------------------------------------------------------------------
    def check_invalid_values(self, rules: dict | None = None) -> dict:
        """
        Checks custom rules for logical (not statistical) invalidity.
        Each rule is a function applied to the entire DataFrame and should return a
        boolean Series (True = invalid).

        If no rules are provided, guesses some default general rules (non-negative numbers,
        reasonable score ranges) based on column names.

        Output: Dictionary {rule_name: DataFrame of violating rows}
        """
        rules = rules or self._default_invalid_rules()
        violations = {}
        for name, rule_fn in rules.items():
            try:
                mask = rule_fn(self.df)
                if mask.any():
                    violations[name] = self.df[mask]
            except Exception as e:
                violations[name] = f"Error executing this rule: {e}"
        return violations

    def _default_invalid_rules(self) -> dict:
        rules = {}
        cols = self.df.columns

        # Scores 1 to 5 should not be outside range (ignore NaN -
        # NaN is a missing value issue, not invalid; check_missing handles it)
        for col in cols:
            if "1to5" in col or "_1_to_5" in col:
                rules[f"{col}_out_of_range"] = (
                    lambda df, c=col: pd.to_numeric(df[c], errors="coerce").notna() &
                                       (pd.to_numeric(df[c], errors="coerce").between(1, 5) == False)
                )

        # Assessment scores (0-100) - same logic: only check non-NaN values
        if "assessment_score" in cols:
            rules["assessment_score_out_of_range"] = (
                lambda df: pd.to_numeric(df["assessment_score"], errors="coerce").notna() &
                           (pd.to_numeric(df["assessment_score"], errors="coerce").between(0, 100) == False)
            )

        # Completion percentages should be between 0 and 100
        for col in cols:
            if "pct" in col.lower() or "percentage" in col.lower():
                rules[f"{col}_out_of_0_100"] = (
                    lambda df, c=col: pd.to_numeric(df[c], errors="coerce").notna() &
                                       (pd.to_numeric(df[c], errors="coerce").between(0, 100) == False)
                )

        # Reasonable age for an employee (range, configurable)
        if "age" in cols:
            rules["age_unrealistic"] = (
                lambda df: pd.to_numeric(df["age"], errors="coerce").notna() &
                           (pd.to_numeric(df["age"], errors="coerce").between(16, 75) == False)
            )

        # goals_achieved should not exceed goals_total
        if "goals_achieved" in cols and "goals_total" in cols:
            rules["goals_achieved_exceeds_total"] = (
                lambda df: pd.to_numeric(df["goals_achieved"], errors="coerce") >
                            pd.to_numeric(df["goals_total"], errors="coerce")
            )

        # Remove rules that remain None (column not found)
        return {k: v for k, v in rules.items() if v is not None}

    # ------------------------------------------------------------------
    # 5) FULL REPORT (readable summary of everything above)
    # ------------------------------------------------------------------
    def full_report(self, outlier_columns: list | None = None,
                     duplicate_subset: list | None = None,
                     invalid_rules: dict | None = None) -> None:
        """Print a readable summary report of all checks on this table."""
        print("=" * 70)
        print(f"Data Quality Report: {self.table_name}  ({len(self.df):,} rows)")
        print("=" * 70)

        # missing
        missing = self.check_missing()
        if missing.empty:
            print("\n[Missing Values] No columns have missing values.")
        else:
            print("\n[Missing Values]")
            print(missing.to_string(index=False))
            unexplained = missing[missing["needs_review"]]
            if not unexplained.empty:
                print(f"\n  -> {len(unexplained)} columns have missing values that need manual review "
                      f"(no logical rule explanation).")

        # duplicates
        dups = self.check_duplicates(subset=duplicate_subset)
        print(f"\n[Duplicates] Found {len(dups)} duplicate rows "
              f"{'(based on ' + str(duplicate_subset) + ')' if duplicate_subset else '(entire row)'}.")

        # outliers
        if outlier_columns:
            outliers = self.check_outliers(outlier_columns)
            print(f"\n[Outliers] {len(outliers)} rows have at least one statistical outlier (IQR) in columns "
                  f"{outlier_columns}.")

        # invalid
        invalid = self.check_invalid_values(invalid_rules)
        if invalid:
            print(f"\n[Invalid / Inconsistent Values]")
            for name, res in invalid.items():
                count = len(res) if isinstance(res, pd.DataFrame) else "N/A"
                print(f"  - {name}: {count} rows")
        else:
            print("\n[Invalid / Inconsistent Values] Nothing found.")

        print("\n" + "=" * 70)


# ============================================================================
# Cross-table check: Referential Integrity
# (Not inside the class because it compares two tables, not a single table)
# ============================================================================
def check_referential_integrity(child_df: pd.DataFrame, parent_df: pd.DataFrame,
                                 fk_col: str, pk_col: str,
                                 fk_table: str = "child", pk_table: str = "parent") -> pd.DataFrame:
    """
    Checks whether every value in fk_col of child_df exists in pk_col of parent_df
    (i.e., whether a join is possible without losing data).

    Output: Rows from child_df whose foreign key is "orphaned"
    (not found in the parent table).
    """
    valid_keys = set(parent_df[pk_col].dropna())
    orphan_mask = ~child_df[fk_col].isin(valid_keys)
    orphans = child_df[orphan_mask]

    print(f"[Referential Integrity] {fk_table}.{fk_col} -> {pk_table}.{pk_col}: "
          f"{len(orphans)} orphan rows (no match in parent table).")
    return orphans