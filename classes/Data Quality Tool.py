"""
Data Quality & Profiling Tool
=================================
ابزار حرفه‌ای تحلیل کیفیت داده و پروفایلینگ Dataset (CSV / Excel / JSON / Parquet)
با گزارش زیبا در ترمینال (Rich) و خروجی HTML مستقل و آفلاین.

نحوه اجرا:
    python data_quality_tool.py path/to/dataset.csv
    python data_quality_tool.py path/to/dataset.xlsx --sheet Sheet1
    python data_quality_tool.py path/to/dataset.csv --clean   # اجرای پاکسازی پس از پروفایلینگ

نیازمندی‌ها:
    pip install pandas numpy scipy rich matplotlib openpyxl
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import traceback
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
from rich.text import Text
from rich import box

warnings.filterwarnings("ignore")
console = Console()

# ----------------------------------------------------------------------------
# ثابت‌ها و تنظیمات
# ----------------------------------------------------------------------------

HIGH_CARDINALITY_RATIO = 0.9        # اگر nunique/n_rows بیشتر از این باشد => احتمالا ID
MIN_ROWS_FOR_STAT_OUTLIER = 20      # حداقل تعداد رکورد برای اجرای روش‌های آماری outlier
LARGE_DATASET_ROWS = 200_000        # آستانه‌ی داده حجیم برای بهینه‌سازی
DEFAULT_SAMPLE_FOR_HEAVY_OPS = 100_000


# ----------------------------------------------------------------------------
# ساختارهای داده نتیجه
# ----------------------------------------------------------------------------

@dataclass
class ColumnIssues:
    name: str
    dtype: str
    inferred_type: str
    missing_count: int = 0
    missing_pct: float = 0.0
    duplicate_value_count: int = 0
    outlier_info: Dict[str, Any] = field(default_factory=dict)
    invalid_values: List[Any] = field(default_factory=list)
    inconsistencies: List[str] = field(default_factory=list)
    type_issues: List[str] = field(default_factory=list)
    is_high_cardinality_id: bool = False
    stats: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


@dataclass
class DatasetReport:
    file_path: str
    n_rows: int = 0
    n_cols: int = 0
    memory_usage_mb: float = 0.0
    load_warnings: List[str] = field(default_factory=list)
    duplicate_rows_count: int = 0
    duplicate_rows_pct: float = 0.0
    columns: Dict[str, ColumnIssues] = field(default_factory=dict)
    overall_missing_pct: float = 0.0
    quality_score: float = 0.0
    quality_score_breakdown: Dict[str, float] = field(default_factory=dict)
    global_notes: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


# ----------------------------------------------------------------------------
# 1) بارگذاری فایل
# ----------------------------------------------------------------------------

def load_dataset(file_path: str, sheet_name: Optional[str] = None) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """
    بارگذاری فایل Dataset با پشتیبانی از csv, xlsx/xls, json, parquet, tsv.
    برمی‌گرداند: (dataframe یا None, لیست هشدارها/خطاها)
    """
    warnings_list: List[str] = []

    if not os.path.exists(file_path):
        warnings_list.append(f"فایل در مسیر '{file_path}' پیدا نشد.")
        return None, warnings_list

    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext == ".csv":
            try:
                df = pd.read_csv(file_path, encoding="utf-8-sig", low_memory=False)
            except UnicodeDecodeError:
                warnings_list.append("رمزگذاری UTF-8 ناموفق بود؛ تلاش با latin-1.")
                df = pd.read_csv(file_path, encoding="latin-1", low_memory=False)
            except pd.errors.ParserError as e:
                warnings_list.append(f"خطای پارس CSV با جداکننده کاما: {e}. تلاش با تشخیص خودکار جداکننده.")
                df = pd.read_csv(file_path, sep=None, engine="python", encoding="utf-8-sig")
        elif ext == ".tsv":
            df = pd.read_csv(file_path, sep="\t", encoding="utf-8-sig", low_memory=False)
        elif ext in (".xlsx", ".xls"):
            if sheet_name:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
            else:
                xls = pd.ExcelFile(file_path)
                if len(xls.sheet_names) > 1:
                    warnings_list.append(
                        f"فایل اکسل شامل {len(xls.sheet_names)} شیت است: {xls.sheet_names}. "
                        f"شیت اول ('{xls.sheet_names[0]}') بارگذاری شد. برای شیت دیگر از --sheet استفاده کنید."
                    )
                df = pd.read_excel(file_path, sheet_name=xls.sheet_names[0])
        elif ext == ".json":
            df = pd.read_json(file_path)
        elif ext == ".parquet":
            df = pd.read_parquet(file_path)
        else:
            warnings_list.append(f"فرمت فایل '{ext}' پشتیبانی نمی‌شود. فرمت‌های مجاز: csv, tsv, xlsx, xls, json, parquet.")
            return None, warnings_list

        if df is None or df.empty:
            warnings_list.append("فایل بارگذاری شد اما هیچ داده‌ای در آن یافت نشد (Dataset خالی است).")
            return None, warnings_list

        # حذف ستون‌های کاملا بی‌نام/Unnamed تکراری از اکسل/CSV
        unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed:")]
        if unnamed_cols and df[unnamed_cols].isna().all().all():
            df = df.drop(columns=unnamed_cols)
            warnings_list.append(f"{len(unnamed_cols)} ستون خالی بدون‌نام حذف شد.")

        return df, warnings_list

    except Exception as e:
        warnings_list.append(f"خطای غیرمنتظره هنگام بارگذاری فایل: {e}")
        return None, warnings_list


# ----------------------------------------------------------------------------
# تشخیص نوع منطقی ستون (Inferred Type)
# ----------------------------------------------------------------------------

def infer_column_type(series: pd.Series) -> str:
    """تشخیص نوع منطقی ستون فراتر از dtype خام پانداس."""
    s = series.dropna()
    if s.empty:
        return "empty"

    if pd.api.types.is_bool_dtype(series):
        return "boolean"

    if pd.api.types.is_numeric_dtype(series):
        unique_vals = s.unique()
        if set(pd.Series(unique_vals).dropna().unique()).issubset({0, 1}):
            return "boolean_numeric"
        return "numeric"

    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    # تلاش برای تشخیص تاریخ در ستون‌های متنی
    sample = s.astype(str).sample(min(50, len(s)), random_state=1)
    try:
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        if parsed.notna().mean() > 0.85:
            return "datetime_like_text"
    except Exception:
        pass

    n_unique = s.nunique()
    n_total = len(s)
    if n_unique <= max(20, int(0.05 * n_total)):
        return "categorical"

    if n_unique / max(n_total, 1) > HIGH_CARDINALITY_RATIO:
        return "identifier_or_high_cardinality"

    return "text"


# ----------------------------------------------------------------------------
# 2) پروفایلینگ کلی
# ----------------------------------------------------------------------------

def profile_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """آمار کلی سطح Dataset."""
    try:
        mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
        return {
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "memory_usage_mb": round(mem_mb, 3),
            "column_names": list(df.columns),
            "dtypes": {c: str(df[c].dtype) for c in df.columns},
        }
    except Exception as e:
        return {"error": f"خطا در پروفایلینگ کلی: {e}"}


# ----------------------------------------------------------------------------
# 3) Missing Values
# ----------------------------------------------------------------------------

def detect_missing(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    شناسایی مقادیر گمشده صریح (NaN/None) و مقادیر شبه‌گمشده رایج
    مانند رشته خالی، 'NA', 'N/A', '-', '?', 'null', 'none'.
    """
    result: Dict[str, Dict[str, Any]] = {}
    placeholder_tokens = {"na", "n/a", "null", "none", "-", "?", "", "unknown", "nan"}

    for col in df.columns:
        try:
            s = df[col]
            explicit_missing = s.isna().sum()

            hidden_missing = 0
            if s.dtype == object:
                normalized = s.dropna().astype(str).str.strip().str.lower()
                hidden_missing = int(normalized.isin(placeholder_tokens).sum())

            total_missing = int(explicit_missing) + hidden_missing
            pct = round((total_missing / len(df)) * 100, 2) if len(df) else 0.0

            result[col] = {
                "explicit_missing": int(explicit_missing),
                "hidden_missing_placeholders": hidden_missing,
                "total_missing": total_missing,
                "missing_pct": pct,
            }
        except Exception as e:
            result[col] = {"error": f"خطا در بررسی missing برای ستون {col}: {e}"}

    return result


# ----------------------------------------------------------------------------
# 4) Duplicate Rows / Values
# ----------------------------------------------------------------------------

def detect_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    """شناسایی ردیف‌های تکراری کامل و مقادیر تکراری در ستون‌های احتمالا کلید."""
    result: Dict[str, Any] = {}
    try:
        dup_mask = df.duplicated(keep=False)
        dup_count = int(df.duplicated(keep="first").sum())
        result["duplicate_rows_count"] = dup_count
        result["duplicate_rows_pct"] = round((dup_count / len(df)) * 100, 2) if len(df) else 0.0
        result["duplicate_row_indices_sample"] = df[dup_mask].index[:10].tolist()
    except Exception as e:
        result["error"] = f"خطا در بررسی ردیف‌های تکراری: {e}"
        result["duplicate_rows_count"] = 0
        result["duplicate_rows_pct"] = 0.0

    per_column: Dict[str, int] = {}
    for col in df.columns:
        try:
            n_unique = df[col].nunique(dropna=True)
            n_non_null = df[col].notna().sum()
            dup_vals = int(n_non_null - n_unique) if n_non_null > n_unique else 0
            per_column[col] = dup_vals
        except Exception:
            per_column[col] = -1  # نامشخص/خطا
    result["duplicate_values_per_column"] = per_column
    return result


# ----------------------------------------------------------------------------
# 5) Outlier Detection (چندروشی)
# ----------------------------------------------------------------------------

def detect_outliers(df: pd.DataFrame, column_types: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """
    تشخیص Outlier فقط برای ستون‌های عددی (numeric) که ID/high-cardinality نیستند.
    روش انتخابی بر اساس حجم داده و شکل توزیع (Skewness):
      - IQR (Tukey Fence): مقاوم، مناسب توزیع‌های نامتقارن و نمونه کوچک/متوسط. همیشه اجرا می‌شود (پایه).
      - Z-Score: مناسب توزیع نزدیک نرمال (|skew| کم) و n >= 20.
      - Modified Z-Score (MAD-based): جایگزین مقاوم‌تر Z-Score برای داده‌های با outlier شدید یا چوله.
      - Isolation Forest: در صورت وجود چند ستون عددی هم‌زمان (تحلیل چندمتغیره) - اختیاری و سبک.
    """
    results: Dict[str, Dict[str, Any]] = {}

    numeric_cols = [c for c, t in column_types.items() if t in ("numeric",)]

    for col in numeric_cols:
        try:
            series = df[col].dropna()
            n = len(series)
            col_result: Dict[str, Any] = {"methods_used": [], "reasoning": []}

            if n < 5:
                col_result["notes"] = "داده کافی برای تشخیص Outlier وجود ندارد (کمتر از ۵ مقدار معتبر)."
                results[col] = col_result
                continue

            skewness = float(series.skew()) if n > 2 else 0.0
            col_result["skewness"] = round(skewness, 3)

            # --- روش ۱: IQR (همیشه، به‌عنوان روش پایه و مقاوم) ---
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                iqr_outliers = pd.Index([])
                col_result["reasoning"].append("IQR=0 (داده تقریبا ثابت) -> روش IQR قابل استفاده معنادار نیست.")
            else:
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                iqr_outliers = series[(series < lower) | (series > upper)].index
                col_result["reasoning"].append(
                    "IQR انتخاب شد چون نسبت به داده‌های چوله (skewed) و outlierهای شدید مقاوم است."
                )
            col_result["methods_used"].append("IQR")
            col_result["iqr_bounds"] = {"lower": float(q1 - 1.5 * iqr) if iqr else None,
                                          "upper": float(q3 + 1.5 * iqr) if iqr else None}
            col_result["iqr_outlier_count"] = len(iqr_outliers)

            # --- روش ۲ یا ۳: بسته به Skewness و حجم نمونه ---
            if n >= MIN_ROWS_FOR_STAT_OUTLIER:
                if abs(skewness) < 1:
                    # نزدیک به نرمال -> Z-Score کلاسیک مناسب است
                    mean, std = series.mean(), series.std()
                    if std and std > 0:
                        z = (series - mean) / std
                        z_outliers = series[np.abs(z) > 3].index
                        col_result["methods_used"].append("Z-Score")
                        col_result["reasoning"].append(
                            "توزیع نزدیک به نرمال است (|skewness|<1)؛ Z-Score (آستانه ۳) مناسب انتخاب شد."
                        )
                        col_result["zscore_outlier_count"] = len(z_outliers)
                    else:
                        col_result["reasoning"].append("انحراف معیار صفر است؛ Z-Score قابل محاسبه نیست.")
                else:
                    # چوله یا دارای دنباله سنگین -> Modified Z-Score (MAD) مقاوم‌تر
                    median = series.median()
                    mad = (series - median).abs().median()
                    if mad and mad > 0:
                        modified_z = 0.6745 * (series - median) / mad
                        mad_outliers = series[np.abs(modified_z) > 3.5].index
                        col_result["methods_used"].append("Modified Z-Score (MAD)")
                        col_result["reasoning"].append(
                            "توزیع چوله است (|skewness|>=1)؛ Modified Z-Score مبتنی بر MAD انتخاب شد چون "
                            "نسبت به outlierهای افراطی حساسیت کمتری نسبت به Z-Score دارد."
                        )
                        col_result["mad_outlier_count"] = len(mad_outliers)
                    else:
                        col_result["reasoning"].append("MAD صفر است (داده متمرکز)؛ Modified Z-Score قابل اعتماد نیست.")
            else:
                col_result["reasoning"].append(
                    f"حجم نمونه ({n}) کمتر از حد آستانه ({MIN_ROWS_FOR_STAT_OUTLIER}) است؛ "
                    "فقط IQR (که به حجم کم داده حساسیت کمتری دارد) اجرا شد."
                )

            # نمونه مقادیر outlier (حداکثر ۵ مورد) از IQR برای نمایش
            if len(iqr_outliers) > 0:
                col_result["sample_outlier_values"] = series.loc[iqr_outliers].head(5).tolist()

            results[col] = col_result

        except Exception as e:
            results[col] = {"error": f"خطا در تشخیص outlier برای ستون {col}: {e}"}

    # --- روش چندمتغیره اختیاری: Isolation Forest (فقط اگر sklearn موجود و چند ستون عددی باشد) ---
    if len(numeric_cols) >= 2:
        try:
            from sklearn.ensemble import IsolationForest
            sub_df = df[numeric_cols].dropna()
            if len(sub_df) >= 20:
                sample_df = sub_df if len(sub_df) <= DEFAULT_SAMPLE_FOR_HEAVY_OPS else sub_df.sample(
                    DEFAULT_SAMPLE_FOR_HEAVY_OPS, random_state=42)
                iso = IsolationForest(contamination="auto", random_state=42, n_jobs=-1)
                preds = iso.fit_predict(sample_df)
                multivariate_outlier_count = int((preds == -1).sum())
                results["__multivariate__"] = {
                    "method": "Isolation Forest",
                    "reasoning": "چند ستون عددی هم‌زمان موجود است؛ Isolation Forest برای تشخیص outlierهای "
                                 "چندمتغیره (ترکیبی) که با روش‌های تک‌متغیره قابل شناسایی نیستند اجرا شد.",
                    "outlier_count": multivariate_outlier_count,
                    "sample_size_used": len(sample_df),
                }
        except ImportError:
            results["__multivariate__"] = {
                "note": "scikit-learn نصب نیست؛ تحلیل چندمتغیره Isolation Forest انجام نشد (اختیاری)."
            }
        except Exception as e:
            results["__multivariate__"] = {"error": f"خطا در Isolation Forest: {e}"}

    return results


# ----------------------------------------------------------------------------
# 6) Invalid Values
# ----------------------------------------------------------------------------

def detect_invalid_values(df: pd.DataFrame, column_types: Dict[str, str]) -> Dict[str, List[str]]:
    """
    شناسایی مقادیر غیرممکن/غیرمنطقی بر اساس نام ستون (heuristic) و قواعد عمومی:
      - اعداد منفی در ستون‌هایی با نام مرتبط با سن/قیمت/مقدار/تعداد
      - سن غیرمنطقی (>120 یا <0)
      - درصد خارج از بازه [0,100]
      - تاریخ در آینده برای ستون‌هایی مثل تولد
    """
    results: Dict[str, List[str]] = {}

    for col in df.columns:
        issues: List[str] = []
        col_lower = str(col).lower()
        try:
            ctype = column_types.get(col)
            if ctype == "numeric":
                series = df[col].dropna()
                if series.empty:
                    continue

                if any(k in col_lower for k in ["age", "سن"]):
                    bad = series[(series < 0) | (series > 120)]
                    if len(bad) > 0:
                        issues.append(f"{len(bad)} مقدار سنی غیرمنطقی (منفی یا بیش از ۱۲۰) یافت شد.")

                if any(k in col_lower for k in ["percent", "درصد", "pct", "rate"]):
                    bad = series[(series < 0) | (series > 100)]
                    if len(bad) > 0:
                        issues.append(f"{len(bad)} مقدار خارج از بازه منطقی درصد [۰,۱۰۰] یافت شد.")

                if any(k in col_lower for k in ["price", "cost", "amount", "quantity", "count",
                                                  "قیمت", "مبلغ", "تعداد", "هزینه"]):
                    bad = series[series < 0]
                    if len(bad) > 0:
                        issues.append(f"{len(bad)} مقدار منفی در ستونی که انتظار می‌رود غیرمنفی باشد یافت شد.")

                inf_count = int(np.isinf(series).sum()) if np.issubdtype(series.dtype, np.number) else 0
                if inf_count > 0:
                    issues.append(f"{inf_count} مقدار بی‌نهایت (inf/-inf) یافت شد.")

            elif ctype in ("datetime", "datetime_like_text"):
                parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
                future_dates = parsed[parsed > pd.Timestamp.now()]
                if any(k in col_lower for k in ["birth", "تولد", "created", "ثبت"]) and len(future_dates) > 0:
                    issues.append(f"{len(future_dates)} تاریخ در آینده برای ستونی مرتبط با تاریخ گذشته یافت شد.")

            elif ctype in ("text", "categorical"):
                series = df[col].dropna().astype(str)
                if any(k in col_lower for k in ["email", "ایمیل"]):
                    invalid_emails = series[~series.str.contains(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", regex=True, na=False)]
                    if len(invalid_emails) > 0:
                        issues.append(f"{len(invalid_emails)} مقدار ایمیل با فرمت نامعتبر یافت شد.")

                if any(k in col_lower for k in ["phone", "mobile", "تلفن", "موبایل"]):
                    digits_only = series.str.replace(r"\D", "", regex=True)
                    invalid_phones = series[(digits_only.str.len() < 8) | (digits_only.str.len() > 15)]
                    if len(invalid_phones) > 0:
                        issues.append(f"{len(invalid_phones)} مقدار شماره تماس با طول غیرمعمول یافت شد.")

            if issues:
                results[col] = issues

        except Exception as e:
            results[col] = [f"خطا در بررسی مقادیر نامعتبر: {e}"]

    return results


# ----------------------------------------------------------------------------
# 7) Inconsistencies
# ----------------------------------------------------------------------------

def detect_inconsistencies(df: pd.DataFrame, column_types: Dict[str, str]) -> Dict[str, List[str]]:
    """
    ناسازگاری‌های رایج:
      - ناسازگاری حروف بزرگ/کوچک یا فاصله اضافی در مقادیر یکسان متنی (مثلا 'Tehran' vs 'tehran ')
      - واحدهای مخلوط در یک ستون عددی که در واقع رشته است (مثلا '10kg' و '10')
      - فرمت‌های متفاوت تاریخ در یک ستون متنی
    """
    results: Dict[str, List[str]] = {}

    for col in df.columns:
        issues: List[str] = []
        ctype = column_types.get(col)
        try:
            if ctype in ("categorical", "text"):
                series = df[col].dropna().astype(str)
                if series.empty:
                    continue
                normalized = series.str.strip().str.lower()
                original_unique = series.nunique()
                normalized_unique = normalized.nunique()
                if normalized_unique < original_unique:
                    diff = original_unique - normalized_unique
                    issues.append(
                        f"{diff} مقدار به دلیل تفاوت حروف بزرگ/کوچک یا فاصله اضافی، عملا تکراری محسوب می‌شوند "
                        f"(مثلا 'Ali' و 'ali ' یکی هستند)."
                    )

            if ctype == "datetime_like_text":
                sample = df[col].dropna().astype(str).head(200)
                patterns = set()
                for val in sample:
                    if "/" in val:
                        patterns.add("slash")
                    elif "-" in val:
                        patterns.add("dash")
                    elif "." in val:
                        patterns.add("dot")
                if len(patterns) > 1:
                    issues.append(f"چند فرمت مختلف تاریخ در یک ستون تشخیص داده شد: {patterns}")

            if issues:
                results[col] = issues
        except Exception as e:
            results[col] = [f"خطا در بررسی ناسازگاری: {e}"]

    return results


# ----------------------------------------------------------------------------
# 8) Data Type Issues
# ----------------------------------------------------------------------------

def detect_data_type_issues(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    شناسایی مواردی مثل:
      - ستون عددی که به اشتباه به صورت متن (object) ذخیره شده (مثلا به دلیل کاراکتر خاص)
      - ستون تاریخ که به صورت متن ذخیره شده
      - وجود مقادیر مختلط از انواع مختلف در یک ستون object
    """
    results: Dict[str, List[str]] = {}

    for col in df.columns:
        issues: List[str] = []
        try:
            series = df[col]
            if series.dtype == object:
                non_null = series.dropna()
                if non_null.empty:
                    continue

                numeric_convertible = pd.to_numeric(non_null, errors="coerce")
                numeric_ratio = numeric_convertible.notna().mean()
                if numeric_ratio > 0.9:
                    issues.append(
                        f"حدود {round(numeric_ratio*100,1)}٪ مقادیر این ستون قابل تبدیل به عدد هستند "
                        "اما ستون به صورت متن (object) ذخیره شده است."
                    )

                date_convertible = pd.to_datetime(non_null, errors="coerce", format="mixed")
                date_ratio = date_convertible.notna().mean()
                if date_ratio > 0.85:
                    issues.append(
                        f"حدود {round(date_ratio*100,1)}٪ مقادیر این ستون قابل تبدیل به تاریخ هستند "
                        "اما ستون به صورت متن ذخیره شده است."
                    )

                types_found = non_null.map(type).nunique()
                if types_found > 1:
                    issues.append(f"این ستون شامل {types_found} نوع پایتونی مختلف در مقادیرش است (نوع مخلوط).")

            if issues:
                results[col] = issues
        except Exception as e:
            results[col] = [f"خطا در بررسی نوع داده: {e}"]

    return results


# ----------------------------------------------------------------------------
# محاسبه Data Quality Score (شفاف و قابل توضیح)
# ----------------------------------------------------------------------------

def calculate_quality_score(report: DatasetReport) -> Tuple[float, Dict[str, float]]:
    """
    امتیاز کیفیت داده از ۰ تا ۱۰۰، بر پایه میانگین وزنی جریمه‌های زیر:
      - Missing (وزن ۳۵٪): درصد کل مقادیر گمشده
      - Duplicate Rows (وزن ۲۰٪): درصد ردیف‌های تکراری
      - Type Issues (وزن ۱۵٪): نسبت ستون‌های دارای مشکل نوع داده
      - Invalid Values (وزن ۱۵٪): نسبت ستون‌های دارای مقدار نامعتبر
      - Inconsistencies (وزن ۱۵٪): نسبت ستون‌های دارای ناسازگاری
    این امتیاز صرفا یک شاخص راهنما است، نه معیار علمی قطعی کیفیت داده.
    """
    breakdown = {}
    n_cols = max(report.n_cols, 1)

    missing_penalty = min(report.overall_missing_pct, 100) * 0.35
    breakdown["missing_penalty"] = round(missing_penalty, 2)

    dup_penalty = min(report.duplicate_rows_pct, 100) * 0.20
    breakdown["duplicate_penalty"] = round(dup_penalty, 2)

    cols_with_type_issues = sum(1 for c in report.columns.values() if c.type_issues)
    type_penalty = (cols_with_type_issues / n_cols) * 100 * 0.15
    breakdown["type_issues_penalty"] = round(type_penalty, 2)

    cols_with_invalid = sum(1 for c in report.columns.values() if c.invalid_values)
    invalid_penalty = (cols_with_invalid / n_cols) * 100 * 0.15
    breakdown["invalid_values_penalty"] = round(invalid_penalty, 2)

    cols_with_inconsistency = sum(1 for c in report.columns.values() if c.inconsistencies)
    inconsistency_penalty = (cols_with_inconsistency / n_cols) * 100 * 0.15
    breakdown["inconsistency_penalty"] = round(inconsistency_penalty, 2)

    total_penalty = sum(breakdown.values())
    score = max(0.0, 100 - total_penalty)
    return round(score, 1), breakdown


# ----------------------------------------------------------------------------
# اجرای کامل تحلیل و ساخت DatasetReport
# ----------------------------------------------------------------------------

def run_full_analysis(df: pd.DataFrame, file_path: str) -> DatasetReport:
    report = DatasetReport(file_path=file_path)
    report.n_rows, report.n_cols = df.shape

    try:
        report.memory_usage_mb = round(df.memory_usage(deep=True).sum() / (1024 ** 2), 3)
    except Exception as e:
        report.errors.append(f"خطا در محاسبه حجم حافظه: {e}")

    if report.n_rows > LARGE_DATASET_ROWS:
        report.global_notes.append(
            f"Dataset حجیم است ({report.n_rows:,} ردیف). برخی عملیات آماری روی نمونه‌ای از داده اجرا می‌شود."
        )

    column_types: Dict[str, str] = {}
    for col in track(df.columns, description="تشخیص نوع منطقی ستون‌ها...", console=console):
        try:
            column_types[col] = infer_column_type(df[col])
        except Exception as e:
            column_types[col] = "unknown"
            report.errors.append(f"خطا در تشخیص نوع ستون {col}: {e}")

    missing_results = detect_missing(df)
    duplicates_result = detect_duplicates(df)
    report.duplicate_rows_count = duplicates_result.get("duplicate_rows_count", 0)
    report.duplicate_rows_pct = duplicates_result.get("duplicate_rows_pct", 0.0)

    outlier_results = detect_outliers(df, column_types)
    invalid_results = detect_invalid_values(df, column_types)
    inconsistency_results = detect_inconsistencies(df, column_types)
    type_issue_results = detect_data_type_issues(df)

    total_missing_all = sum(v.get("total_missing", 0) for v in missing_results.values() if isinstance(v, dict))
    report.overall_missing_pct = round(
        (total_missing_all / (report.n_rows * report.n_cols)) * 100, 2
    ) if report.n_rows and report.n_cols else 0.0

    for col in df.columns:
        ctype = column_types.get(col, "unknown")
        col_issue = ColumnIssues(name=col, dtype=str(df[col].dtype), inferred_type=ctype)

        m = missing_results.get(col, {})
        col_issue.missing_count = m.get("total_missing", 0)
        col_issue.missing_pct = m.get("missing_pct", 0.0)

        col_issue.duplicate_value_count = duplicates_result.get("duplicate_values_per_column", {}).get(col, 0)

        if ctype == "identifier_or_high_cardinality":
            col_issue.is_high_cardinality_id = True
            col_issue.notes.append("این ستون احتمالا شناسه (ID) یا کاردینالیتی بالا دارد؛ تحلیل آماری روی آن اجرا نشد.")
        else:
            if col in outlier_results:
                col_issue.outlier_info = outlier_results[col]
            if col in invalid_results:
                col_issue.invalid_values = invalid_results[col]
            if col in inconsistency_results:
                col_issue.inconsistencies = inconsistency_results[col]

        if col in type_issue_results:
            col_issue.type_issues = type_issue_results[col]

        try:
            if ctype == "numeric":
                s = df[col].dropna()
                if len(s) > 0:
                    col_issue.stats = {
                        "mean": round(float(s.mean()), 3),
                        "median": round(float(s.median()), 3),
                        "std": round(float(s.std()), 3) if len(s) > 1 else 0.0,
                        "min": float(s.min()),
                        "max": float(s.max()),
                    }
            elif ctype in ("categorical", "text"):
                vc = df[col].value_counts(dropna=True).head(5)
                col_issue.stats = {"top_values": vc.to_dict(), "n_unique": int(df[col].nunique(dropna=True))}
        except Exception as e:
            report.errors.append(f"خطا در محاسبه آمار توصیفی برای ستون {col}: {e}")

        report.columns[col] = col_issue

    if "__multivariate__" in outlier_results:
        report.global_notes.append(json.dumps(outlier_results["__multivariate__"], ensure_ascii=False))

    report.quality_score, report.quality_score_breakdown = calculate_quality_score(report)
    return report


# ----------------------------------------------------------------------------
# نمایش گزارش در ترمینال با Rich
# ----------------------------------------------------------------------------

def render_rich_report(report: DatasetReport) -> None:
    console.rule("[bold cyan]گزارش تحلیل کیفیت داده[/bold cyan]")

    summary_table = Table(title="خلاصه Dataset", box=box.ROUNDED, show_lines=False)
    summary_table.add_column("ویژگی", style="bold yellow")
    summary_table.add_column("مقدار", style="white")
    summary_table.add_row("مسیر فایل", report.file_path)
    summary_table.add_row("تعداد ردیف‌ها", f"{report.n_rows:,}")
    summary_table.add_row("تعداد ستون‌ها", str(report.n_cols))
    summary_table.add_row("حجم حافظه (MB)", str(report.memory_usage_mb))
    summary_table.add_row("ردیف‌های تکراری", f"{report.duplicate_rows_count} ({report.duplicate_rows_pct}%)")
    summary_table.add_row("درصد کلی Missing", f"{report.overall_missing_pct}%")
    console.print(summary_table)

    score_color = "green" if report.quality_score >= 80 else "yellow" if report.quality_score >= 50 else "red"
    console.print(Panel(
        f"[bold {score_color}]{report.quality_score} / 100[/bold {score_color}]\n\n"
        f"تفکیک جریمه‌ها: {report.quality_score_breakdown}",
        title="Data Quality Score (شاخص راهنما، نه معیار قطعی)",
        border_style=score_color,
    ))

    col_table = Table(title="بررسی ستون به ستون", box=box.MINIMAL_DOUBLE_HEAD, show_lines=True)
    for header in ["ستون", "نوع", "Missing%", "Outlier", "Invalid", "Inconsistency", "Type Issue", "توضیح"]:
        col_table.add_column(header, overflow="fold")

    for name, c in report.columns.items():
        outlier_summary = ""
        if c.is_high_cardinality_id:
            outlier_summary = "—"
        elif c.outlier_info:
            outlier_summary = str(c.outlier_info.get("iqr_outlier_count", "-"))
        col_table.add_row(
            name,
            c.inferred_type,
            f"{c.missing_pct}%",
            outlier_summary,
            "بله" if c.invalid_values else "-",
            "بله" if c.inconsistencies else "-",
            "بله" if c.type_issues else "-",
            "شناسه/کاردینالیتی بالا" if c.is_high_cardinality_id else "",
        )
    console.print(col_table)

    if report.global_notes:
        console.print(Panel("\n".join(report.global_notes), title="یادداشت‌های کلی", border_style="blue"))
    if report.errors:
        console.print(Panel("\n".join(report.errors), title="خطاهای رخ‌داده در حین تحلیل", border_style="red"))

    console.rule("[bold cyan]پایان گزارش ترمینال[/bold cyan]")


# ----------------------------------------------------------------------------
# 9) تولید گزارش HTML
# ----------------------------------------------------------------------------

def _make_chart_base64(report: DatasetReport) -> Optional[str]:
    """ساخت نمودار میله‌ای درصد Missing به ازای هر ستون، برگرداندن base64 برای embed در HTML."""
    if not MATPLOTLIB_AVAILABLE:
        return None
    try:
        cols = list(report.columns.keys())
        missing_pcts = [report.columns[c].missing_pct for c in cols]
        if not cols:
            return None

        fig, ax = plt.subplots(figsize=(max(6, len(cols) * 0.5), 4))
        ax.bar(cols, missing_pcts, color="#e67e22")
        ax.set_ylabel("درصد Missing")
        ax.set_title("درصد مقادیر گمشده به ازای ستون")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110)
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    except Exception:
        return None


def generate_html_report(report: DatasetReport, output_path: str) -> Optional[str]:
    """تولید فایل HTML مستقل (بدون نیاز به اینترنت) شامل گزارش کامل."""
    try:
        chart_b64 = _make_chart_base64(report)
        chart_html = (
            f'<img src="data:image/png;base64,{chart_b64}" alt="نمودار Missing Values" style="max-width:100%;border-radius:8px;">'
            if chart_b64 else "<p><em>نمودار در دسترس نیست (matplotlib نصب نیست یا ستونی وجود ندارد).</em></p>"
        )

        score_color = "#2ecc71" if report.quality_score >= 80 else "#f39c12" if report.quality_score >= 50 else "#e74c3c"

        col_rows = ""
        for name, c in report.columns.items():
            outlier_txt = "—"
            if c.is_high_cardinality_id:
                outlier_txt = "تحلیل نشد (ID/کاردینالیتی بالا)"
            elif c.outlier_info:
                parts = []
                if "iqr_outlier_count" in c.outlier_info:
                    parts.append(f"IQR: {c.outlier_info['iqr_outlier_count']}")
                if "zscore_outlier_count" in c.outlier_info:
                    parts.append(f"Z-Score: {c.outlier_info['zscore_outlier_count']}")
                if "mad_outlier_count" in c.outlier_info:
                    parts.append(f"MAD: {c.outlier_info['mad_outlier_count']}")
                outlier_txt = " | ".join(parts) if parts else "بدون outlier قابل توجه"

            reasoning_txt = ""
            if c.outlier_info.get("reasoning"):
                reasoning_txt = "<br><small>" + " / ".join(c.outlier_info["reasoning"]) + "</small>"

            invalid_txt = "<br>".join(c.invalid_values) if c.invalid_values else "-"
            inconsistency_txt = "<br>".join(c.inconsistencies) if c.inconsistencies else "-"
            type_issue_txt = "<br>".join(c.type_issues) if c.type_issues else "-"

            stats_txt = ""
            if c.stats:
                stats_txt = "<br>".join(f"{k}: {v}" for k, v in c.stats.items())

            col_rows += f"""
            <tr>
                <td><b>{name}</b><br><small>{c.dtype} → {c.inferred_type}</small></td>
                <td>{c.missing_count} ({c.missing_pct}%)</td>
                <td>{outlier_txt}{reasoning_txt}</td>
                <td>{invalid_txt}</td>
                <td>{inconsistency_txt}</td>
                <td>{type_issue_txt}</td>
                <td><small>{stats_txt}</small></td>
            </tr>"""

        notes_html = "".join(f"<li>{n}</li>" for n in report.global_notes) or "<li>موردی ثبت نشده است.</li>"
        errors_html = "".join(f"<li>{e}</li>" for e in report.errors) or "<li>خطایی رخ نداد.</li>"

        html = f"""<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
<meta charset="UTF-8">
<title>گزارش کیفیت داده</title>
<style>
  body {{ font-family: Tahoma, "Vazirmatn", sans-serif; background:#f5f6fa; color:#2c3e50; margin:0; padding:0; }}
  .container {{ max-width: 1100px; margin: 30px auto; padding: 20px; }}
  header {{ background: linear-gradient(135deg,#2c3e50,#34495e); color:#fff; padding:30px; border-radius:12px; }}
  header h1 {{ margin:0 0 8px 0; }}
  header p {{ margin:0; opacity:0.85; }}
  .cards {{ display:flex; gap:16px; margin:24px 0; flex-wrap:wrap; }}
  .card {{ background:#fff; border-radius:10px; padding:18px 22px; box-shadow:0 2px 8px rgba(0,0,0,0.06); flex:1; min-width:180px; }}
  .card h3 {{ margin:0 0 6px 0; font-size:14px; color:#7f8c8d; }}
  .card .value {{ font-size:26px; font-weight:bold; }}
  .score-box {{ background:#fff; border-radius:10px; padding:20px; margin:20px 0; box-shadow:0 2px 8px rgba(0,0,0,0.06); }}
  .score-value {{ font-size:42px; font-weight:bold; color:{score_color}; }}
  table {{ width:100%; border-collapse: collapse; background:#fff; border-radius:10px; overflow:hidden; margin:20px 0; }}
  th, td {{ padding:10px 12px; text-align:right; border-bottom:1px solid #ecf0f1; font-size:13px; vertical-align:top; }}
  th {{ background:#2c3e50; color:#fff; }}
  tr:hover {{ background:#f9fbfd; }}
  section {{ background:#fff; border-radius:10px; padding:20px; margin:20px 0; box-shadow:0 2px 8px rgba(0,0,0,0.06); }}
  section h2 {{ border-right:4px solid #e67e22; padding-right:10px; }}
  .footer {{ text-align:center; color:#95a5a6; font-size:12px; margin:30px 0; }}
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>گزارش تحلیل کیفیت داده</h1>
    <p>فایل: {os.path.basename(report.file_path)} | تولید شده در: {report.generated_at}</p>
  </header>

  <div class="cards">
    <div class="card"><h3>تعداد ردیف‌ها</h3><div class="value">{report.n_rows:,}</div></div>
    <div class="card"><h3>تعداد ستون‌ها</h3><div class="value">{report.n_cols}</div></div>
    <div class="card"><h3>حجم حافظه</h3><div class="value">{report.memory_usage_mb} MB</div></div>
    <div class="card"><h3>ردیف‌های تکراری</h3><div class="value">{report.duplicate_rows_count} ({report.duplicate_rows_pct}%)</div></div>
    <div class="card"><h3>Missing کلی</h3><div class="value">{report.overall_missing_pct}%</div></div>
  </div>

  <div class="score-box">
    <h2>Data Quality Score</h2>
    <div class="score-value">{report.quality_score} / 100</div>
    <p><small>محاسبه شده بر اساس جریمه‌های وزنی: Missing (۳۵٪)، Duplicate Rows (۲۰٪)، Type Issues (۱۵٪)،
    Invalid Values (۱۵٪)، Inconsistencies (۱۵٪). این عدد صرفا یک شاخص راهنماست، نه معیار علمی قطعی.</small></p>
    <p><small>جزئیات: {report.quality_score_breakdown}</small></p>
  </div>

  <section>
    <h2>نمودار مقادیر گمشده</h2>
    {chart_html}
  </section>

  <section>
    <h2>تحلیل ستون به ستون</h2>
    <table>
      <tr><th>ستون / نوع</th><th>Missing</th><th>Outlier</th><th>Invalid Values</th><th>Inconsistencies</th><th>Type Issues</th><th>آمار توصیفی</th></tr>
      {col_rows}
    </table>
  </section>

  <section>
    <h2>یادداشت‌های کلی و پیشنهادات</h2>
    <ul>{notes_html}</ul>
  </section>

  <section>
    <h2>خطاهای رخ‌داده در حین تحلیل</h2>
    <ul>{errors_html}</ul>
  </section>

  <div class="footer">تولید شده توسط ابزار Data Quality &amp; Profiling Tool</div>
</div>
</body>
</html>"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        return output_path

    except Exception as e:
        console.print(f"[bold red]خطا در تولید فایل HTML: {e}[/bold red]")
        return None


# ----------------------------------------------------------------------------
# Cleaning (فقط در صورت درخواست صریح کاربر با --clean)
# ----------------------------------------------------------------------------

def clean_dataset(df: pd.DataFrame, report: DatasetReport) -> pd.DataFrame:
    """
    پاکسازی پایه‌ی داده بر اساس یافته‌های گزارش:
      - حذف ردیف‌های تکراری کامل
      - این تابع محافظه‌کارانه عمل می‌کند و مقادیر outlier/missing را به صورت خودکار حذف نمی‌کند
        مگر اینکه کاربر منطق دقیق‌تری بخواهد (قابل توسعه).
    """
    cleaned = df.copy()
    try:
        before = len(cleaned)
        cleaned = cleaned.drop_duplicates(keep="first")
        after = len(cleaned)
        console.print(f"[green]{before - after} ردیف تکراری حذف شد.[/green]")
    except Exception as e:
        console.print(f"[red]خطا در حذف ردیف‌های تکراری: {e}[/red]")
    return cleaned


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Data Quality & Profiling Tool")
    parser.add_argument("file_path", help="مسیر فایل Dataset (csv, xlsx, xls, json, parquet, tsv)")
    parser.add_argument("--sheet", default=None, help="نام شیت اکسل (اختیاری)")
    parser.add_argument("--clean", action="store_true", help="اجرای پاکسازی پایه پس از تحلیل")
    parser.add_argument("--output", default=None, help="مسیر خروجی HTML (اختیاری)")
    args = parser.parse_args()

    try:
        df, load_warnings = load_dataset(args.file_path, args.sheet)
        for w in load_warnings:
            console.print(f"[yellow]⚠ {w}[/yellow]")

        if df is None:
            console.print("[bold red]تحلیل متوقف شد: بارگذاری فایل ناموفق بود.[/bold red]")
            sys.exit(1)

        report = run_full_analysis(df, args.file_path)
        report.load_warnings = load_warnings

        render_rich_report(report)

        output_path = args.output or (
            f"data_quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        )
        result_path = generate_html_report(report, output_path)
        if result_path:
            console.print(Panel(
                f"[bold green]فایل HTML با موفقیت ساخته شد:[/bold green]\n{os.path.abspath(result_path)}",
                border_style="green"
            ))

        if args.clean:
            cleaned_df = clean_dataset(df, report)
            cleaned_path = os.path.splitext(args.file_path)[0] + "_cleaned.csv"
            cleaned_df.to_csv(cleaned_path, index=False, encoding="utf-8-sig")
            console.print(Panel(
                f"[bold green]فایل پاکسازی‌شده ذخیره شد:[/bold green]\n{os.path.abspath(cleaned_path)}",
                border_style="green"
            ))

    except KeyboardInterrupt:
        console.print("\n[yellow]عملیات توسط کاربر لغو شد.[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[bold red]خطای پیش‌بینی‌نشده: {e}[/bold red]")
        console.print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()