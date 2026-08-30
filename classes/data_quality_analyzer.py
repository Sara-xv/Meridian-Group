#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Quality Analyzer & Cleaning Toolkit
==========================================
ابزار حرفه‌ای تحلیل کیفیت داده و شناسایی مشکلات Dataset.

قابلیت‌ها:
    - بارگذاری خودکار CSV / Excel / JSON / Parquet / TSV
    - پروفایلینگ کلی Dataset (شکل، حافظه، انواع داده)
    - شناسایی Missing Values با چند الگو (NaN, None, "", "NA", "-", ...)
    - شناسایی Duplicate Rows و Duplicate Values در ستون‌های کلیدی
    - شناسایی Outlier با چند روش (IQR, Z-Score, Modified Z-Score, Isolation-ready)
      به‌صورت خودکار بر اساس توزیع و حجم داده انتخاب می‌شود
    - شناسایی Invalid Values (بر اساس قواعد منطقی نوع داده: سن منفی، درصد خارج از بازه و ...)
    - شناسایی Inconsistencies (فرمت‌های ناهمگون رشته‌ای، case mismatch، فاصله اضافه)
    - شناسایی Data Type Issues (ستون عددی که به صورت رشته ذخیره شده و ...)
    - محاسبه Data Quality Score شفاف و قابل تفکیک
    - گزارش زیبا در ترمینال با Rich
    - تولید گزارش HTML مستقل (بدون نیاز به اینترنت یا اجرای مجدد پایتون)
    - Cleaning اختیاری، فقط با درخواست صریح کاربر (اجرا نمی‌شود مگر فراخوانی شود)

نویسنده: تولید شده توسط Claude برای Sara (Sorenix.data)
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import traceback
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import track
    from rich.text import Text
    from rich import box
    from rich.rule import Rule
    RICH_AVAILABLE = True
except ImportError:  # pragma: no cover
    RICH_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    MATPLOTLIB_AVAILABLE = False


# ==========================================================================
# ثابت‌ها و پیکربندی
# ==========================================================================

# رشته‌هایی که معمولاً معادل Missing هستند اما pandas آن‌ها را NaN تشخیص نمی‌دهد
EXTRA_NA_TOKENS = {
    "", " ", "na", "n/a", "n.a.", "n\\a", "null", "none", "nil", "-", "--",
    "?", "unknown", "unk", "not available", "not applicable", "nan",
    "#n/a", "#null!", "missing", "empty", "undefined", "()", "[]",
}

# آستانه‌ها و پارامترهای پیش‌فرض (قابل تغییر هنگام ساخت آبجکت)
DEFAULT_CONFIG = {
    "high_cardinality_ratio": 0.95,   # اگر تعداد مقادیر یکتا / تعداد ردیف بیشتر از این باشد -> شناسه محتمل
    "high_cardinality_min_rows": 20,  # حداقل تعداد ردیف برای اعمال قانون بالا (روی دیتای خیلی کوچک بی‌معنی است)
    "id_name_patterns": [r"\bid$", r"^id\b", r"_id$", r"^id_", r"uuid", r"guid", r"identifier", r"code$"],
    "outlier_small_n_threshold": 30,      # زیر این تعداد، روش‌های پارامتریک قابل‌اتکا نیستند
    "outlier_large_n_threshold": 50_000,  # بالای این تعداد، برای کارایی از نمونه‌گیری در برخی محاسبات کمکی استفاده می‌شود
    "skew_threshold_for_iqr": 1.0,     # چولگی بیش از این مقدار -> ترجیح روش‌های مقاوم (IQR / Modified Z)
    "zscore_threshold": 3.0,
    "modified_zscore_threshold": 3.5,
    "iqr_multiplier": 1.5,
    "duplicate_sample_limit": 20,      # حداکثر تعداد نمونه ردیف تکراری که در گزارش نمایش داده می‌شود
    "max_categories_display": 15,      # حداکثر تعداد دسته که در فراوانی ستون‌های دسته‌ای نمایش داده می‌شود
}

SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls", ".json", ".parquet", ".txt"}


@dataclass
class LoadResult:
    """نتیجه بارگذاری فایل، شامل داده و متادیتای مربوط به نحوه خواندن آن."""
    success: bool
    dataframe: Optional[pd.DataFrame] = None
    file_path: Optional[str] = None
    file_format: Optional[str] = None
    sheet_name: Optional[str] = None
    load_warnings: list = field(default_factory=list)
    error_message: Optional[str] = None


def _normalize_na_string(value: Any) -> bool:
    """بررسی می‌کند آیا یک مقدار رشته‌ای معادل رایج Missing Value است یا نه."""
    if not isinstance(value, str):
        return False
    return value.strip().lower() in EXTRA_NA_TOKENS


class DatasetLoader:
    """
    مسئول بارگذاری امن Dataset از فرمت‌های رایج.
    از هرگونه خطا (فایل نبودن، انکودینگ اشتباه، شیت نبودن و ...) به شکل قابل‌فهم عبور می‌کند
    و هرگز برنامه را متوقف نمی‌کند؛ در عوض LoadResult با success=False برمی‌گرداند.
    """

    ENCODINGS_TO_TRY = ["utf-8", "utf-8-sig", "cp1256", "latin1", "utf-16"]
    SEPARATORS_TO_TRY = [",", ";", "\t", "|"]

    @staticmethod
    def load(file_path: Union[str, Path], sheet_name: Optional[Union[str, int]] = None) -> LoadResult:
        path = Path(file_path)
        load_warnings: list = []

        if not path.exists():
            return LoadResult(
                success=False,
                file_path=str(path),
                error_message=f"فایل در مسیر «{path}» پیدا نشد. لطفاً مسیر یا نام فایل را بررسی کنید.",
            )

        if path.stat().st_size == 0:
            return LoadResult(
                success=False,
                file_path=str(path),
                error_message="فایل خالی است (حجم صفر بایت) و قابل تحلیل نیست.",
            )

        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            return LoadResult(
                success=False,
                file_path=str(path),
                error_message=(
                    f"فرمت «{ext}» پشتیبانی نمی‌شود. فرمت‌های مجاز: "
                    f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}"
                ),
            )

        try:
            if ext in (".xlsx", ".xls"):
                return DatasetLoader._load_excel(path, sheet_name, load_warnings)
            elif ext == ".json":
                return DatasetLoader._load_json(path, load_warnings)
            elif ext == ".parquet":
                df = pd.read_parquet(path)
                return LoadResult(True, df, str(path), "parquet", load_warnings=load_warnings)
            else:  # csv / tsv / txt
                return DatasetLoader._load_delimited(path, load_warnings)
        except Exception as exc:  # noqa: BLE001
            return LoadResult(
                success=False,
                file_path=str(path),
                error_message=f"خطای غیرمنتظره در بارگذاری فایل: {exc}",
            )

    @staticmethod
    def _load_delimited(path: Path, load_warnings: list) -> LoadResult:
        last_error = None
        for encoding in DatasetLoader.ENCODINGS_TO_TRY:
            try:
                # اگر پسوند tsv است مستقیم با تب بخوان
                if path.suffix.lower() == ".tsv":
                    df = pd.read_csv(path, sep="\t", encoding=encoding, engine="python",
                                      na_values=list(EXTRA_NA_TOKENS), keep_default_na=True,
                                      on_bad_lines="warn")
                    if encoding != "utf-8":
                        load_warnings.append(f"فایل با انکودینگ «{encoding}» خوانده شد (utf-8 جواب نداد).")
                    return LoadResult(True, df, str(path), "tsv", load_warnings=load_warnings)

                # حالت CSV/TXT: جداکننده را حدس بزن
                sep = DatasetLoader._sniff_separator(path, encoding)
                df = pd.read_csv(
                    path, sep=sep, encoding=encoding, engine="python",
                    na_values=list(EXTRA_NA_TOKENS), keep_default_na=True,
                    on_bad_lines="warn",
                )
                if df.shape[1] == 1 and sep != ",":
                    load_warnings.append(
                        f"جداکننده «{sep}» تشخیص داده شد اما فقط یک ستون تولید شد؛ "
                        "لطفاً از صحت فرمت فایل مطمئن شوید."
                    )
                if encoding != "utf-8":
                    load_warnings.append(f"فایل با انکودینگ «{encoding}» خوانده شد (utf-8 جواب نداد).")
                return LoadResult(True, df, str(path), "csv", load_warnings=load_warnings)
            except (UnicodeDecodeError, UnicodeError) as exc:
                last_error = exc
                continue
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue

        return LoadResult(
            success=False,
            file_path=str(path),
            error_message=(
                "بارگذاری فایل متنی/CSV با هیچ‌کدام از انکودینگ‌های رایج موفق نشد. "
                f"آخرین خطا: {last_error}"
            ),
        )

    @staticmethod
    def _sniff_separator(path: Path, encoding: str) -> str:
        """حدس جداکننده بر اساس اولین چند خط فایل."""
        try:
            with open(path, "r", encoding=encoding, errors="strict") as f:
                sample_lines = [f.readline() for _ in range(5)]
            sample = "".join(sample_lines)
            counts = {sep: sample.count(sep) for sep in DatasetLoader.SEPARATORS_TO_TRY}
            best_sep = max(counts, key=counts.get)
            return best_sep if counts[best_sep] > 0 else ","
        except Exception:  # noqa: BLE001
            return ","

    @staticmethod
    def _load_excel(path: Path, sheet_name: Optional[Union[str, int]], load_warnings: list) -> LoadResult:
        try:
            xls = pd.ExcelFile(path)
        except Exception as exc:  # noqa: BLE001
            return LoadResult(
                success=False, file_path=str(path),
                error_message=f"فایل Excel قابل باز شدن نیست (ممکن است خراب یا رمزگذاری‌شده باشد): {exc}",
            )

        available_sheets = xls.sheet_names
        target_sheet = sheet_name if sheet_name is not None else available_sheets[0]

        if isinstance(target_sheet, str) and target_sheet not in available_sheets:
            return LoadResult(
                success=False, file_path=str(path),
                error_message=(
                    f"شیت «{target_sheet}» در فایل پیدا نشد. شیت‌های موجود: {available_sheets}"
                ),
            )

        if len(available_sheets) > 1 and sheet_name is None:
            load_warnings.append(
                f"فایل شامل {len(available_sheets)} شیت است ({available_sheets})؛ "
                f"شیت اول «{available_sheets[0]}» به صورت خودکار انتخاب شد."
            )

        try:
            df = pd.read_excel(xls, sheet_name=target_sheet, na_values=list(EXTRA_NA_TOKENS))
        except Exception as exc:  # noqa: BLE001
            return LoadResult(
                success=False, file_path=str(path),
                error_message=f"خطا در خواندن شیت «{target_sheet}»: {exc}",
            )

        return LoadResult(True, df, str(path), "excel", sheet_name=str(target_sheet), load_warnings=load_warnings)

    @staticmethod
    def _load_json(path: Path, load_warnings: list) -> LoadResult:
        try:
            df = pd.read_json(path)
            return LoadResult(True, df, str(path), "json", load_warnings=load_warnings)
        except ValueError:
            # شاید JSON Lines باشد یا ساختار nested
            try:
                df = pd.read_json(path, lines=True)
                load_warnings.append("فایل به‌صورت JSON Lines خوانده شد.")
                return LoadResult(True, df, str(path), "json", load_warnings=load_warnings)
            except Exception:
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        raw = json.load(f)
                    df = pd.json_normalize(raw)
                    load_warnings.append("ساختار JSON تودرتو بود؛ با json_normalize مسطح شد.")
                    return LoadResult(True, df, str(path), "json", load_warnings=load_warnings)
                except Exception as exc:  # noqa: BLE001
                    return LoadResult(
                        success=False, file_path=str(path),
                        error_message=f"ساختار JSON قابل تبدیل به جدول نیست: {exc}",
                    )
        except Exception as exc:  # noqa: BLE001
            return LoadResult(
                success=False, file_path=str(path),
                error_message=f"خطا در خواندن فایل JSON: {exc}",
            )


# ==========================================================================
# تشخیص نوع منطقی ستون‌ها
# ==========================================================================

class ColumnKind:
    NUMERIC = "numeric"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    CATEGORICAL = "categorical"
    TEXT_FREEFORM = "text_freeform"
    IDENTIFIER = "identifier"
    CONSTANT = "constant"
    EMPTY = "empty"


class ColumnTypeDetector:
    """
    تشخیص نوع «منطقی» هر ستون، فراتر از dtype خام pandas.
    مثلاً ستونی با dtype=object که شامل اعداد است، numeric تشخیص داده می‌شود؛
    یا ستونی int که در واقع شناسه (ID) است، از تحلیل آماری کنار گذاشته می‌شود.
    """

    def __init__(self, config: dict):
        self.config = config

    def detect(self, series: pd.Series, column_name: str) -> dict:
        n_total = len(series)
        non_null = series.dropna()
        n_non_null = len(non_null)

        result = {
            "kind": ColumnKind.TEXT_FREEFORM,
            "reason": "",
            "is_high_cardinality": False,
            "coerced_numeric": None,   # نسخه‌ی numeric-coerced اگر ستون رشته‌ای بود ولی محتوای عددی داشت
            "coerced_datetime": None,
        }

        if n_non_null == 0:
            result["kind"] = ColumnKind.EMPTY
            result["reason"] = "تمام مقادیر این ستون خالی/Missing هستند."
            return result

        n_unique = non_null.nunique()

        if n_unique == 1:
            result["kind"] = ColumnKind.CONSTANT
            result["reason"] = f"تمام مقادیر غیرخالی این ستون برابر «{non_null.iloc[0]}» هستند."
            return result

        # ۱) بررسی شناسه بودن بر اساس نام ستون
        name_lower = str(column_name).lower().strip()
        looks_like_id_name = any(re.search(pat, name_lower) for pat in self.config["id_name_patterns"])

        # ۲) بررسی Cardinality بالا (فقط وقتی تعداد ردیف کافی باشد)
        cardinality_ratio = n_unique / n_non_null if n_non_null else 0
        high_cardinality = (
            n_total >= self.config["high_cardinality_min_rows"]
            and cardinality_ratio >= self.config["high_cardinality_ratio"]
        )
        result["is_high_cardinality"] = high_cardinality

        if looks_like_id_name and (high_cardinality or n_unique == n_non_null):
            result["kind"] = ColumnKind.IDENTIFIER
            result["reason"] = (
                f"نام ستون الگوی شناسه دارد و Cardinality آن {cardinality_ratio:.0%} است "
                f"({n_unique} مقدار یکتا از {n_non_null})."
            )
            return result

        # ۳) بولین
        if pd.api.types.is_bool_dtype(series):
            result["kind"] = ColumnKind.BOOLEAN
            result["reason"] = "نوع پایه ستون Boolean است."
            return result

        unique_vals_lower = set()
        if non_null.dtype == object or pd.api.types.is_string_dtype(non_null):
            try:
                unique_vals_lower = set(str(v).strip().lower() for v in non_null.unique()[:50])
            except Exception:  # noqa: BLE001
                unique_vals_lower = set()

        bool_like_sets = [
            {"true", "false"}, {"yes", "no"}, {"y", "n"}, {"0", "1"},
            {"t", "f"}, {"male", "female"} if False else set(),
        ]
        if n_unique <= 2 and unique_vals_lower and any(
            unique_vals_lower.issubset(s) for s in bool_like_sets if s
        ):
            result["kind"] = ColumnKind.BOOLEAN
            result["reason"] = f"فقط دو مقدار یکتا دارد که به‌صورت boolean قابل تفسیرند: {unique_vals_lower}"
            return result

        # ۴) عددی — یا native، یا رشته‌ای قابل تبدیل به عدد
        if pd.api.types.is_numeric_dtype(series):
            if looks_like_id_name and high_cardinality:
                result["kind"] = ColumnKind.IDENTIFIER
                result["reason"] = "نام ستون الگوی شناسه دارد و Cardinality بالاست."
            else:
                result["kind"] = ColumnKind.NUMERIC
                result["reason"] = "نوع پایه ستون عددی است."
            return result

        if non_null.dtype == object or pd.api.types.is_string_dtype(non_null):
            coerced = pd.to_numeric(non_null.astype(str).str.strip(), errors="coerce")
            success_ratio = coerced.notna().mean()
            if success_ratio >= 0.9:
                result["kind"] = ColumnKind.NUMERIC
                result["coerced_numeric"] = coerced
                result["reason"] = (
                    f"ستون به صورت متنی ذخیره شده اما {success_ratio:.0%} مقادیر آن قابل تبدیل به عدد هستند "
                    "(احتمالاً مشکل Data Type)."
                )
                return result

        # ۵) تاریخ
        if pd.api.types.is_datetime64_any_dtype(series):
            result["kind"] = ColumnKind.DATETIME
            result["reason"] = "نوع پایه ستون تاریخ/زمان است."
            return result

        if non_null.dtype == object or pd.api.types.is_string_dtype(non_null):
            sample = non_null.astype(str).str.strip()
            sample_for_test = sample.sample(min(200, len(sample)), random_state=1) if len(sample) > 200 else sample
            date_like_pattern = re.compile(
                r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}"
                r"|^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}"
                r"|^\d{4}\d{2}\d{2}$"
            )
            plausible = sample_for_test.str.match(date_like_pattern)
            if plausible.mean() >= 0.85:
                try:
                    coerced_dt = pd.to_datetime(sample, errors="coerce", format="mixed")
                    if coerced_dt.notna().mean() >= 0.85:
                        result["kind"] = ColumnKind.DATETIME
                        result["coerced_datetime"] = coerced_dt
                        result["reason"] = "ستون به صورت متنی ذخیره شده اما الگوی تاریخ دارد (مشکل Data Type)."
                        return result
                except Exception:  # noqa: BLE001
                    pass

        # ۶) شناسه بر اساس Cardinality صرف (حتی بدون نام مشکوک)، وقتی عملاً هر مقدار یکتاست
        if high_cardinality and n_unique == n_non_null and n_non_null >= self.config["high_cardinality_min_rows"]:
            result["kind"] = ColumnKind.IDENTIFIER
            result["reason"] = (
                f"هر مقدار این ستون یکتاست ({n_unique}/{n_non_null})؛ به‌احتمال زیاد شناسه یا متن آزاد کاملاً منحصربه‌فرد است."
            )
            return result

        # ۷) دسته‌ای در برابر متن آزاد
        avg_len = non_null.astype(str).str.len().mean() if n_non_null else 0
        if n_unique <= max(30, int(0.2 * n_non_null)) and avg_len <= 40:
            result["kind"] = ColumnKind.CATEGORICAL
            result["reason"] = f"تعداد دسته‌های محدود ({n_unique}) و طول متوسط کوتاه ({avg_len:.0f} کاراکتر)."
        else:
            result["kind"] = ColumnKind.TEXT_FREEFORM
            result["reason"] = f"تنوع بالا ({n_unique} مقدار یکتا) یا طول متوسط بلند ({avg_len:.0f} کاراکتر)؛ متن آزاد است."

        return result


# ==========================================================================
# پروفایلینگ کلی Dataset
# ==========================================================================

class DatasetProfiler:
    """پروفایلینگ سطح-Dataset: شکل، حافظه، نوع ستون‌ها، آمار کلی."""

    def __init__(self, df: pd.DataFrame, config: dict):
        self.df = df
        self.config = config
        self.type_detector = ColumnTypeDetector(config)

    def profile(self) -> dict:
        df = self.df
        n_rows, n_cols = df.shape

        memory_bytes = df.memory_usage(deep=True).sum()

        column_profiles = {}
        for col in df.columns:
            try:
                column_profiles[col] = self._profile_column(df[col], col)
            except Exception as exc:  # noqa: BLE001
                column_profiles[col] = {
                    "kind": ColumnKind.TEXT_FREEFORM,
                    "reason": f"خطا در تشخیص نوع ستون: {exc}",
                    "is_high_cardinality": False,
                    "coerced_numeric": None,
                    "coerced_datetime": None,
                    "error": str(exc),
                }

        kind_counts: dict = {}
        for prof in column_profiles.values():
            kind_counts[prof["kind"]] = kind_counts.get(prof["kind"], 0) + 1

        return {
            "n_rows": n_rows,
            "n_cols": n_cols,
            "memory_bytes": memory_bytes,
            "memory_human": self._human_readable_bytes(memory_bytes),
            "column_profiles": column_profiles,
            "kind_counts": kind_counts,
            "dtypes": {col: str(dt) for col, dt in df.dtypes.items()},
            "fully_empty_columns": [c for c, p in column_profiles.items() if p["kind"] == ColumnKind.EMPTY],
            "constant_columns": [c for c, p in column_profiles.items() if p["kind"] == ColumnKind.CONSTANT],
            "identifier_columns": [c for c, p in column_profiles.items() if p["kind"] == ColumnKind.IDENTIFIER],
        }

    def _profile_column(self, series: pd.Series, col_name: str) -> dict:
        return self.type_detector.detect(series, col_name)

    @staticmethod
    def _human_readable_bytes(n_bytes: float) -> str:
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if n_bytes < 1024:
                return f"{n_bytes:.2f} {unit}"
            n_bytes /= 1024
        return f"{n_bytes:.2f} PB"


# ==========================================================================
# شناسایی Missing Values
# ==========================================================================

class MissingValueDetector:
    """
    شناسایی مقادیر گمشده، هم به‌صورت NaN واقعی و هم رشته‌های معادل آن
    (که ممکن است در بارگذاری اولیه به NaN تبدیل نشده باشند).
    """

    def __init__(self, config: dict):
        self.config = config

    def detect(self, df: pd.DataFrame) -> dict:
        try:
            n_rows = len(df)
            if n_rows == 0:
                return {"error": "Dataset خالی است؛ محاسبه Missing Values ممکن نیست.", "per_column": {}, "total_missing_cells": 0}

            per_column = {}
            total_missing = 0

            for col in df.columns:
                series = df[col]
                native_na_mask = series.isna()
                # رشته‌های معادل NA که ممکن است هنوز به صورت متن باقی مانده باشند
                extra_na_mask = series.apply(_normalize_na_string) if series.dtype == object else pd.Series(False, index=series.index)
                combined_mask = native_na_mask | extra_na_mask

                n_missing = int(combined_mask.sum())
                total_missing += n_missing

                per_column[col] = {
                    "n_missing": n_missing,
                    "pct_missing": (n_missing / n_rows * 100) if n_rows else 0.0,
                    "n_native_nan": int(native_na_mask.sum()),
                    "n_masked_string_na": int(extra_na_mask.sum()) if series.dtype == object else 0,
                }

            rows_with_any_missing = df.isna().any(axis=1).sum()
            rows_fully_missing = df.isna().all(axis=1).sum()

            # الگوی missingness: آیا missing بودن یک ستون با ستون دیگر همبستگی دارد؟ (کمک به تشخیص MNAR/MAR ساده)
            missing_correlation_notes = self._detect_missingness_correlation(df)

            total_cells = n_rows * max(len(df.columns), 1)
            return {
                "per_column": per_column,
                "total_missing_cells": total_missing,
                "total_cells": total_cells,
                "overall_missing_pct": (total_missing / total_cells * 100) if total_cells else 0.0,
                "rows_with_any_missing": int(rows_with_any_missing),
                "rows_fully_missing": int(rows_fully_missing),
                "missing_correlation_notes": missing_correlation_notes,
                "columns_sorted_by_missing": sorted(
                    per_column.items(), key=lambda kv: kv[1]["n_missing"], reverse=True
                ),
            }
        except Exception as exc:  # noqa: BLE001
            return {"error": f"خطا در تحلیل Missing Values: {exc}", "per_column": {}, "total_missing_cells": 0}

    def _detect_missingness_correlation(self, df: pd.DataFrame, max_cols: int = 15) -> list:
        """بررسی می‌کند آیا missing بودن دو ستون با هم هم‌رخداد بالایی دارند (نشانه الگوی سیستماتیک)."""
        notes = []
        try:
            na_df = df.isna()
            cols_with_missing = [c for c in df.columns if na_df[c].any()]
            if len(cols_with_missing) < 2:
                return notes
            cols_with_missing = cols_with_missing[:max_cols]  # برای کارایی
            na_sub = na_df[cols_with_missing].astype(int)
            corr = na_sub.corr()
            seen = set()
            for c1 in corr.columns:
                for c2 in corr.columns:
                    if c1 == c2 or (c2, c1) in seen:
                        continue
                    seen.add((c1, c2))
                    val = corr.loc[c1, c2]
                    if pd.notna(val) and val >= 0.6:
                        notes.append(
                            f"الگوی هم‌رخدادی: مقادیر گمشده در «{c1}» و «{c2}» همبستگی بالایی دارند "
                            f"(r={val:.2f})؛ احتمالاً علت مشترکی دارند."
                        )
        except Exception:  # noqa: BLE001
            pass
        return notes


# ==========================================================================
# شناسایی Duplicate Rows / Values
# ==========================================================================

class DuplicateDetector:
    """شناسایی ردیف‌های تکراری کامل و مقادیر تکراری در ستون‌هایی که باید یکتا باشند."""

    def __init__(self, config: dict):
        self.config = config

    def detect(self, df: pd.DataFrame, identifier_columns: Optional[list] = None) -> dict:
        try:
            n_rows = len(df)
            if n_rows == 0:
                return {"error": "Dataset خالی است.", "n_duplicate_rows": 0}

            full_dup_mask = df.duplicated(keep=False)
            n_full_dup_rows = int(full_dup_mask.sum())
            n_full_dup_groups = int(df[full_dup_mask].drop_duplicates().shape[0]) if n_full_dup_rows else 0

            sample_limit = self.config["duplicate_sample_limit"]
            sample_indices = df[full_dup_mask].index.tolist()[:sample_limit]

            result = {
                "n_duplicate_rows": n_full_dup_rows,
                "pct_duplicate_rows": (n_full_dup_rows / n_rows * 100) if n_rows else 0.0,
                "n_duplicate_groups": n_full_dup_groups,
                "sample_duplicate_indices": sample_indices,
                "identifier_duplicate_issues": [],
            }

            # اگر ستون‌های شناسه داریم، هرگونه تکرار در آن‌ها یک مشکل جدی‌تر است (نقض یکتایی مورد انتظار)
            if identifier_columns:
                for col in identifier_columns:
                    if col not in df.columns:
                        continue
                    col_dup_mask = df[col].duplicated(keep=False) & df[col].notna()
                    n_col_dups = int(col_dup_mask.sum())
                    if n_col_dups > 0:
                        result["identifier_duplicate_issues"].append({
                            "column": col,
                            "n_duplicated_values": n_col_dups,
                            "n_unique_duplicated": int(df.loc[col_dup_mask, col].nunique()),
                            "sample_values": df.loc[col_dup_mask, col].unique()[:10].tolist(),
                        })

            return result
        except Exception as exc:  # noqa: BLE001
            return {"error": f"خطا در تحلیل Duplicate ها: {exc}", "n_duplicate_rows": 0}


# ==========================================================================
# شناسایی Outlier — چندروشی و انتخاب هوشمند بر اساس توزیع/حجم داده
# ==========================================================================

class OutlierDetector:
    """
    شناسایی داده‌های پرت با چند روش مکمل. روش‌ها صرفاً روی ستون‌های عددی
    (native یا coerced) اجرا می‌شوند و هرگز روی متن/دسته/تاریخ/شناسه اجرا نمی‌گردد.

    منطق انتخاب روش:
        - IQR (Tukey's Fences): مقاوم در برابر چولگی و outlier های شدید؛ همیشه محاسبه می‌شود
          چون هیچ فرض توزیعی ندارد و ارزان است.
        - Z-Score: فقط وقتی داده تقریباً نرمال است (چولگی کم) و حجم نمونه کافی (>=۳۰) باشد
          دقیق است؛ در غیر این صورت با outlier های خودش (میانگین/انحراف معیار حساس به outlier)
          گمراه‌کننده می‌شود.
        - Modified Z-Score (بر پایه MAD - Median Absolute Deviation): جایگزین مقاوم Z-Score
          برای داده‌های چوله یا دارای outlier شدید، چون از میانه/MAD به‌جای میانگین/std استفاده می‌کند.
        - در نمونه‌های خیلی کوچک (<۳۰) هیچ روش پارامتریکی قابل‌اتکا نیست؛ فقط IQR با احتیاط
          گزارش و به کاربر هشدار داده می‌شود.

    نتیجه‌ی نهایی هر ستون شامل outlier هایی است که حداقل با یک روش شناسایی شده‌اند،
    به همراه اینکه هر ایندکس با کدام روش(ها) پرچم خورده (برای شفافیت).
    """

    def __init__(self, config: dict):
        self.config = config

    def detect(self, df: pd.DataFrame, column_profiles: dict) -> dict:
        results = {}
        for col, prof in column_profiles.items():
            if prof["kind"] != ColumnKind.NUMERIC:
                continue  # طبق محدودیت: outlier فقط برای ستون‌های عددی
            try:
                series = prof["coerced_numeric"] if prof.get("coerced_numeric") is not None else df[col]
                series = pd.to_numeric(series, errors="coerce")
                results[col] = self._detect_for_column(series)
            except Exception as exc:  # noqa: BLE001
                results[col] = {"error": f"خطا در تحلیل Outlier ستون «{col}»: {exc}"}
        return results

    def _detect_for_column(self, series: pd.Series) -> dict:
        clean = series.dropna()
        n = len(clean)

        if n == 0:
            return {"error": "پس از حذف مقادیر گمشده، داده‌ای برای تحلیل باقی نماند.", "methods_used": []}

        if clean.nunique() == 1:
            return {
                "n_outliers": 0, "methods_used": [], "outlier_indices": [],
                "note": "تمام مقادیر یکسان هستند؛ مفهوم Outlier در اینجا معنا ندارد.",
            }

        small_n = n < self.config["outlier_small_n_threshold"]
        skewness = float(stats.skew(clean)) if n >= 3 else 0.0
        high_skew = abs(skewness) >= self.config["skew_threshold_for_iqr"]

        methods_used = []
        flags = {}  # index -> set(method names)

        # --- روش ۱: IQR (Tukey's Fences) — همیشه محاسبه می‌شود ---
        q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            k = self.config["iqr_multiplier"]
            lower_bound, upper_bound = q1 - k * iqr, q3 + k * iqr
            iqr_outliers = clean[(clean < lower_bound) | (clean > upper_bound)]
            methods_used.append("IQR")
            for idx in iqr_outliers.index:
                flags.setdefault(idx, set()).add("IQR")
            iqr_info = {
                "lower_bound": float(lower_bound), "upper_bound": float(upper_bound),
                "n_flagged": int(len(iqr_outliers)),
            }
        else:
            iqr_info = {"note": "IQR صفر است (بیش از ۷۵٪ داده یک مقدار ثابت دارند)؛ این روش قابل اعمال نیست."}

        # --- روش ۲: Z-Score — فقط اگر نمونه کافی و چولگی پایین باشد ---
        zscore_info = {"applied": False}
        if not small_n and not high_skew and clean.std(ddof=0) > 0:
            z = (clean - clean.mean()) / clean.std(ddof=0)
            threshold = self.config["zscore_threshold"]
            z_outliers = clean[z.abs() > threshold]
            methods_used.append("Z-Score")
            for idx in z_outliers.index:
                flags.setdefault(idx, set()).add("Z-Score")
            zscore_info = {
                "applied": True, "threshold": threshold, "n_flagged": int(len(z_outliers)),
                "reason_applied": "حجم نمونه کافی (≥30) و چولگی پایین (|skew|<1) است؛ فرض نرمالیتی تقریباً برقرار است.",
            }
        else:
            reason_skipped = []
            if small_n:
                reason_skipped.append(f"حجم نمونه کم است (n={n} < 30)")
            if high_skew:
                reason_skipped.append(f"چولگی بالاست (skew={skewness:.2f})")
            zscore_info["reason_skipped"] = " و ".join(reason_skipped) + "؛ Z-Score با میانگین/انحراف‌معیار حساس به outlier گمراه‌کننده می‌شود."

        # --- روش ۳: Modified Z-Score (MAD-based) — مقاوم، مناسب چولگی بالا یا outlier شدید ---
        median = clean.median()
        mad = float((clean - median).abs().median())
        modified_zscore_info = {"applied": False}
        if mad > 0:
            modified_z = 0.6745 * (clean - median) / mad
            threshold = self.config["modified_zscore_threshold"]
            mz_outliers = clean[modified_z.abs() > threshold]
            methods_used.append("Modified Z-Score (MAD)")
            for idx in mz_outliers.index:
                flags.setdefault(idx, set()).add("Modified Z-Score (MAD)")
            modified_zscore_info = {
                "applied": True, "threshold": threshold, "n_flagged": int(len(mz_outliers)),
                "reason_applied": (
                    "بر پایه میانه و MAD است که نسبت به outlier ها و چولگی مقاوم‌اند؛ "
                    "به‌عنوان روش تکمیلی/جایگزین Z-Score همیشه محاسبه می‌شود."
                ),
            }
        else:
            modified_zscore_info["note"] = "MAD صفر است (بیش از نیمی از داده یک مقدار ثابت دارند)؛ این روش قابل اعمال نیست."

        n_outliers = len(flags)
        consensus_outliers = [idx for idx, methods in flags.items() if len(methods) >= 2]

        selection_reasoning = self._build_reasoning(n, small_n, skewness, high_skew)

        outlier_records = []
        for idx, methods in sorted(flags.items(), key=lambda kv: kv[0])[:50]:  # حداکثر ۵۰ نمونه برای گزارش
            outlier_records.append({
                "index": idx, "value": float(clean.loc[idx]), "methods": sorted(methods),
            })

        return {
            "n_valid_values": n,
            "skewness": skewness,
            "is_high_skew": high_skew,
            "is_small_sample": small_n,
            "methods_used": methods_used,
            "iqr": iqr_info,
            "zscore": zscore_info,
            "modified_zscore": modified_zscore_info,
            "n_outliers": n_outliers,
            "pct_outliers": (n_outliers / n * 100) if n else 0.0,
            "n_consensus_outliers": len(consensus_outliers),
            "outlier_records_sample": outlier_records,
            "selection_reasoning": selection_reasoning,
            "min_value": float(clean.min()),
            "max_value": float(clean.max()),
            "mean": float(clean.mean()),
            "median": float(median),
            "std": float(clean.std(ddof=0)),
        }

    @staticmethod
    def _build_reasoning(n: int, small_n: bool, skewness: float, high_skew: bool) -> str:
        parts = [f"حجم داده معتبر: {n}."]
        if small_n:
            parts.append(
                "چون حجم نمونه کم است (<۳۰)، نتایج آماری (به‌ویژه Z-Score) قابل‌اتکا نیستند؛ "
                "IQR و Modified Z-Score به‌عنوان روش‌های اصلی در نظر گرفته شدند."
            )
        parts.append(f"چولگی (Skewness) = {skewness:.2f} → {'توزیع نامتقارن است' if high_skew else 'توزیع نسبتاً متقارن است'}.")
        if high_skew:
            parts.append("به‌دلیل چولگی بالا، Z-Score رد شد و بر IQR/Modified Z-Score (مقاوم‌تر) تکیه شد.")
        else:
            parts.append("چولگی پایین است، بنابراین Z-Score نیز به‌عنوان روش مکمل معتبر اجرا شد.")
        return " ".join(parts)


# ==========================================================================
# شناسایی Invalid Values (بر اساس قواعد منطقی)
# ==========================================================================

class InvalidValueDetector:
    """
    شناسایی مقادیری که آماری «پرت» نیستند لزوماً، اما از نظر منطقی/دامنه‌ای غیرممکن‌اند.
    قواعد بر اساس نام ستون (heuristic) به‌صورت خودکار اعمال می‌شوند و همیشه در گزارش
    توضیح داده می‌شود که چرا یک قاعده روی یک ستون اجرا شده است (شفافیت کامل).

    این لایه از Outlier Detection مستقل است: یک مقدار می‌تواند outlier آماری نباشد
    اما invalid باشد (مثل rating=6 وقتی مقیاس ۱ تا ۵ است) یا برعکس.
    """

    # الگوهای نام ستون -> (تابع اعتبارسنجی، توضیح قاعده)
    RULE_PATTERNS = [
        (r"age$|^age", lambda s: (s < 0) | (s > 120), "سن نباید منفی یا بیشتر از ۱۲۰ باشد."),
        (r"percent|percentage|_pct$|^pct", lambda s: (s < 0) | (s > 100), "درصد باید بین ۰ تا ۱۰۰ باشد."),
        (r"rate$|ratio$", lambda s: (s < 0) | (s > 1), "نرخ/نسبت معمولاً باید بین ۰ تا ۱ باشد (در صورت مقیاس متفاوت، نادیده بگیرید)."),
        (r"rating$|score$|^rating|^score", lambda s: (s < 0), "امتیاز/نمره نباید منفی باشد."),
        (r"salary|income|wage|price|cost|amount|revenue|payment", lambda s: (s < 0), "مقادیر مالی نباید منفی باشند."),
        (r"quantity|qty|count$|^count|stock", lambda s: (s < 0), "تعداد/موجودی نباید منفی باشد."),
        (r"duration|hours|days|minutes|seconds|_time$", lambda s: (s < 0), "مدت‌زمان نباید منفی باشد."),
        (r"latitude$|^lat$", lambda s: (s < -90) | (s > 90), "عرض جغرافیایی باید بین -۹۰ تا ۹۰ باشد."),
        (r"longitude$|^lon$|^lng$", lambda s: (s < -180) | (s > 180), "طول جغرافیایی باید بین -۱۸۰ تا ۱۸۰ باشد."),
        (r"month$|^month", lambda s: (s < 1) | (s > 12), "ماه باید بین ۱ تا ۱۲ باشد."),
        (r"day$|^day", lambda s: (s < 1) | (s > 31), "روز ماه باید بین ۱ تا ۳۱ باشد."),
    ]

    def __init__(self, config: dict):
        self.config = config

    def detect(self, df: pd.DataFrame, column_profiles: dict) -> dict:
        results = {}

        for col, prof in column_profiles.items():
            if prof["kind"] != ColumnKind.NUMERIC:
                continue
            try:
                series = prof["coerced_numeric"] if prof.get("coerced_numeric") is not None else df[col]
                series = pd.to_numeric(series, errors="coerce")
                col_result = self._check_column(col, series)
                if col_result is not None:
                    results[col] = col_result
            except Exception as exc:  # noqa: BLE001
                results[col] = {"error": f"خطا در بررسی Invalid Values ستون «{col}»: {exc}"}

        # بررسی تاریخ‌های آینده‌دار غیرمنطقی یا خیلی قدیمی برای ستون‌های datetime
        for col, prof in column_profiles.items():
            if prof["kind"] != ColumnKind.DATETIME:
                continue
            try:
                dt_result = self._check_datetime_column(col, df, prof)
                if dt_result is not None:
                    results[col] = dt_result
            except Exception as exc:  # noqa: BLE001
                results[col] = {"error": f"خطا در بررسی تاریخ ستون «{col}»: {exc}"}

        return results

    def _check_column(self, col_name: str, series: pd.Series) -> Optional[dict]:
        name_lower = str(col_name).lower().strip()
        clean = series.dropna()
        if clean.empty:
            return None

        matched_rules = []
        for pattern, rule_fn, description in self.RULE_PATTERNS:
            if re.search(pattern, name_lower):
                try:
                    mask = rule_fn(clean)
                except Exception:  # noqa: BLE001
                    continue
                n_invalid = int(mask.sum())
                if n_invalid > 0:
                    sample = clean[mask].head(10).tolist()
                    matched_rules.append({
                        "rule": description,
                        "n_invalid": n_invalid,
                        "pct_invalid": n_invalid / len(clean) * 100,
                        "sample_values": sample,
                        "sample_indices": clean[mask].index.tolist()[:10],
                    })

        if not matched_rules:
            return None

        return {"matched_rules": matched_rules, "n_total_flagged": sum(r["n_invalid"] for r in matched_rules)}

    def _check_datetime_column(self, col_name: str, df: pd.DataFrame, prof: dict) -> Optional[dict]:
        series = prof.get("coerced_datetime")
        if series is None:
            series = pd.to_datetime(df[col_name], errors="coerce", format="mixed")
        clean = series.dropna()
        if clean.empty:
            return None

        now = pd.Timestamp.now()
        future_mask = clean > now
        very_old_mask = clean < pd.Timestamp("1900-01-01")

        matched_rules = []
        if future_mask.sum() > 0:
            matched_rules.append({
                "rule": "تاریخ در آینده است (بزرگ‌تر از زمان حال)، که برای بسیاری از زمینه‌ها (تولد، ثبت‌نام گذشته) غیرمنطقی است.",
                "n_invalid": int(future_mask.sum()),
                "pct_invalid": float(future_mask.mean() * 100),
                "sample_values": clean[future_mask].astype(str).head(10).tolist(),
                "sample_indices": clean[future_mask].index.tolist()[:10],
            })
        if very_old_mask.sum() > 0:
            matched_rules.append({
                "rule": "تاریخ پیش از سال ۱۹۰۰ است که معمولاً نشانه خطای ورود داده یا مقدار پیش‌فرض اشتباه است.",
                "n_invalid": int(very_old_mask.sum()),
                "pct_invalid": float(very_old_mask.mean() * 100),
                "sample_values": clean[very_old_mask].astype(str).head(10).tolist(),
                "sample_indices": clean[very_old_mask].index.tolist()[:10],
            })

        if not matched_rules:
            return None
        return {"matched_rules": matched_rules, "n_total_flagged": sum(r["n_invalid"] for r in matched_rules)}


# ==========================================================================
# شناسایی Inconsistencies (ناسازگاری‌های فرمتی/متنی)
# ==========================================================================

class InconsistencyDetector:
    """
    شناسایی ناسازگاری در نمایش داده‌های یکسان از نظر معنایی:
        - اختلاف حروف بزرگ/کوچک (Sales vs sales vs SALES)
        - فاصله‌های اضافی ابتدا/انتها
        - نمایش‌های متفاوت boolean (True/False/yes/no/1/0)
        - فرمت‌های ناهمگون تاریخ در یک ستون رشته‌ای
        - نمایش‌های متفاوت واحد/فرمت عدد (مثلاً "1,000" vs "1000")
    """

    def __init__(self, config: dict):
        self.config = config

    def detect(self, df: pd.DataFrame, column_profiles: dict) -> dict:
        results = {}
        for col, prof in column_profiles.items():
            if prof["kind"] not in (ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN, ColumnKind.TEXT_FREEFORM):
                continue
            try:
                col_result = self._check_text_column(df[col], prof)
                if col_result is not None:
                    results[col] = col_result
            except Exception as exc:  # noqa: BLE001
                results[col] = {"error": f"خطا در بررسی ناسازگاری ستون «{col}»: {exc}"}
        return results

    def _check_text_column(self, series: pd.Series, prof: dict) -> Optional[dict]:
        non_null = series.dropna()
        if non_null.empty or non_null.dtype != object and not pd.api.types.is_string_dtype(non_null):
            # ممکن است bool واقعی یا عددی باشد؛ برای categorical رشته‌ای ادامه بده وگرنه رد کن
            if not pd.api.types.is_object_dtype(non_null) and not pd.api.types.is_string_dtype(non_null):
                return None

        issues = []
        str_series = non_null.astype(str)

        # ۱) case-insensitive duplicates: مقادیر متفاوت که فقط در حروف بزرگ/کوچک فرق دارند
        normalized = str_series.str.strip().str.lower()
        case_groups: dict = {}
        for original, norm in zip(str_series, normalized):
            case_groups.setdefault(norm, set()).add(original)

        case_conflicts = {k: v for k, v in case_groups.items() if len(v) > 1}
        if case_conflicts:
            examples = list(case_conflicts.items())[:5]
            issues.append({
                "type": "case_mismatch",
                "description": "چند نمایش متفاوت (از نظر بزرگی/کوچکی حروف یا فاصله) برای یک مقدار معنایی یکسان وجود دارد.",
                "n_groups_affected": len(case_conflicts),
                "examples": [{"normalized": k, "variants": sorted(v)} for k, v in examples],
            })

        # ۲) فاصله‌های اضافی ابتدا/انتها
        has_padding = (str_series != str_series.str.strip())
        n_padding = int(has_padding.sum())
        if n_padding > 0:
            issues.append({
                "type": "leading_trailing_whitespace",
                "description": "برخی مقادیر دارای فاصله‌ی اضافی در ابتدا/انتها هستند.",
                "n_affected": n_padding,
                "examples": [repr(v) for v in str_series[has_padding].unique()[:5]],
            })

        # ۳) نمایش ناهمگون boolean در ستون‌هایی که عملاً باینری‌اند
        norm_unique = set(normalized.unique())
        boolean_token_groups = [
            {"true", "yes", "y", "1", "t"},
            {"false", "no", "n", "0", "f"},
        ]
        matches_true = norm_unique & boolean_token_groups[0]
        matches_false = norm_unique & boolean_token_groups[1]
        if len(matches_true) > 1 or len(matches_false) > 1:
            issues.append({
                "type": "inconsistent_boolean_representation",
                "description": "این ستون به نظر باینری/بولین می‌آید اما با چند نمایش متفاوت ذخیره شده است (مثلاً هم True/False و هم yes/no).",
                "true_variants": sorted(matches_true),
                "false_variants": sorted(matches_false),
            })

        if not issues:
            return None
        return {"issues": issues}


# ==========================================================================
# شناسایی مشکلات Data Type
# ==========================================================================

class DataTypeIssueDetector:
    """
    شناسایی ستون‌هایی که dtype خام آن‌ها با محتوای واقعی‌شان همخوانی ندارد:
        - ستون عددی که به‌صورت متن (object) ذخیره شده
        - ستون تاریخ که به‌صورت متن ذخیره شده
        - ستون‌های عددی float که در واقع باید int باشند (مثل شمارش) اما به‌خاطر NaN به float تبدیل شده‌اند
        - ستون‌های Mixed-Type (ترکیب انواع مختلف در یک ستون object)
    """

    def __init__(self, config: dict):
        self.config = config

    def detect(self, df: pd.DataFrame, column_profiles: dict) -> dict:
        results = {}
        for col, prof in column_profiles.items():
            try:
                col_issues = []
                raw_dtype = str(df[col].dtype)

                if prof["kind"] == ColumnKind.NUMERIC and prof.get("coerced_numeric") is not None:
                    col_issues.append({
                        "issue": "numeric_stored_as_text",
                        "description": (
                            f"ستون از نظر محتوا عددی است اما با dtype «{raw_dtype}» (متنی) ذخیره شده است. "
                            "توصیه: تبدیل به نوع عددی برای امکان محاسبات و مقایسه صحیح."
                        ),
                    })

                if prof["kind"] == ColumnKind.DATETIME and prof.get("coerced_datetime") is not None:
                    col_issues.append({
                        "issue": "date_stored_as_text",
                        "description": (
                            f"ستون از نظر محتوا تاریخ است اما با dtype «{raw_dtype}» (متنی) ذخیره شده است. "
                            "توصیه: تبدیل به datetime برای امکان مرتب‌سازی و محاسبات زمانی صحیح."
                        ),
                    })

                if prof["kind"] == ColumnKind.NUMERIC and raw_dtype.startswith("float"):
                    non_null = df[col].dropna()
                    if not non_null.empty:
                        numeric_vals = pd.to_numeric(non_null, errors="coerce").dropna()
                        if not numeric_vals.empty and (numeric_vals % 1 == 0).all():
                            n_missing = df[col].isna().sum()
                            col_issues.append({
                                "issue": "float_should_be_int",
                                "description": (
                                    "تمام مقادیر غیرخالی این ستون float عدد صحیح هستند (مثل 5.0، 10.0)؛ "
                                    f"احتمالاً به‌دلیل وجود {n_missing} مقدار گمشده به float تبدیل شده "
                                    "(محدودیت ذاتی pandas/NumPy). پس از رفع Missing Values قابل تبدیل به int است."
                                ),
                            })

                if raw_dtype == "object":
                    non_null = df[col].dropna()
                    if not non_null.empty:
                        type_names = non_null.map(lambda v: type(v).__name__).unique()
                        if len(type_names) > 1:
                            col_issues.append({
                                "issue": "mixed_python_types",
                                "description": f"ستون شامل چند نوع پایتونی مختلف است: {sorted(type_names)}.",
                            })

                if col_issues:
                    results[col] = {"raw_dtype": raw_dtype, "issues": col_issues}
            except Exception as exc:  # noqa: BLE001
                results[col] = {"error": f"خطا در بررسی Data Type ستون «{col}»: {exc}"}
        return results


# ==========================================================================
# محاسبه Data Quality Score
# ==========================================================================

class QualityScoreCalculator:
    """
    محاسبه یک امتیاز کلی (۰ تا ۱۰۰) به‌عنوان «شاخص راهنما» نه معیار علمی قطعی.
    روش محاسبه کاملاً شفاف است: از ۱۰۰ شروع می‌شود و به ازای هر دسته مشکل،
    بر اساس شدت و گستردگی آن، جریمه (Penalty) با سقف مشخص کسر می‌شود.

    هدف این امتیاز، مقایسه نسبی و ردیابی بهبود در طول زمان است؛ نه قضاوت مطلق
    درباره «خوب یا بد بودن» یک Dataset، چون اهمیت هر مشکل به Context کسب‌وکار بستگی دارد.
    """

    # سهم هر دسته از ۱۰۰ امتیاز (جمعاً برابر با حداکثر جریمه ممکن)
    WEIGHTS = {
        "missing": 25,
        "duplicates": 15,
        "outliers": 15,
        "invalid": 20,
        "inconsistency": 15,
        "data_type": 10,
    }

    def calculate(self, n_rows: int, n_cols: int, missing_result: dict, duplicate_result: dict,
                  outlier_result: dict, invalid_result: dict, inconsistency_result: dict,
                  datatype_result: dict) -> dict:
        breakdown = {}

        # Missing: بر اساس درصد کلی سلول‌های گمشده
        missing_pct = missing_result.get("overall_missing_pct", 0.0)
        missing_penalty = min(self.WEIGHTS["missing"], (missing_pct / 20) * self.WEIGHTS["missing"])
        breakdown["missing"] = {
            "penalty": round(missing_penalty, 2),
            "max_penalty": self.WEIGHTS["missing"],
            "basis": f"{missing_pct:.2f}% کل سلول‌ها گمشده‌اند (هر ۲۰٪ ≈ کل سهم این دسته).",
        }

        # Duplicates: بر اساس درصد ردیف‌های تکراری
        dup_pct = duplicate_result.get("pct_duplicate_rows", 0.0)
        dup_penalty = min(self.WEIGHTS["duplicates"], (dup_pct / 10) * self.WEIGHTS["duplicates"])
        breakdown["duplicates"] = {
            "penalty": round(dup_penalty, 2),
            "max_penalty": self.WEIGHTS["duplicates"],
            "basis": f"{dup_pct:.2f}% ردیف‌ها تکراری‌اند (هر ۱۰٪ ≈ کل سهم این دسته).",
        }

        # Outliers: میانگین درصد outlier در ستون‌های عددی
        outlier_pcts = [v.get("pct_outliers", 0.0) for v in outlier_result.values() if "error" not in v]
        avg_outlier_pct = sum(outlier_pcts) / len(outlier_pcts) if outlier_pcts else 0.0
        outlier_penalty = min(self.WEIGHTS["outliers"], (avg_outlier_pct / 15) * self.WEIGHTS["outliers"])
        breakdown["outliers"] = {
            "penalty": round(outlier_penalty, 2),
            "max_penalty": self.WEIGHTS["outliers"],
            "basis": f"میانگین {avg_outlier_pct:.2f}% مقادیر ستون‌های عددی به‌عنوان Outlier پرچم خورده‌اند.",
        }

        # Invalid values: تعداد کل مقادیر نامعتبر نسبت به کل سلول‌ها
        total_invalid = sum(v.get("n_total_flagged", 0) for v in invalid_result.values() if "error" not in v)
        total_cells = max(n_rows * n_cols, 1)
        invalid_pct = total_invalid / total_cells * 100
        invalid_penalty = min(self.WEIGHTS["invalid"], (invalid_pct / 5) * self.WEIGHTS["invalid"])
        breakdown["invalid"] = {
            "penalty": round(invalid_penalty, 2),
            "max_penalty": self.WEIGHTS["invalid"],
            "basis": f"{total_invalid} مقدار غیرمنطقی/غیرممکن شناسایی شد ({invalid_pct:.2f}% کل سلول‌ها).",
        }

        # Inconsistency: تعداد ستون‌های دارای مشکل ناسازگاری نسبت به کل ستون‌ها
        n_inconsistent_cols = len(inconsistency_result)
        inconsistency_ratio = n_inconsistent_cols / max(n_cols, 1)
        inconsistency_penalty = min(self.WEIGHTS["inconsistency"], inconsistency_ratio * self.WEIGHTS["inconsistency"] * 2)
        breakdown["inconsistency"] = {
            "penalty": round(inconsistency_penalty, 2),
            "max_penalty": self.WEIGHTS["inconsistency"],
            "basis": f"{n_inconsistent_cols} از {n_cols} ستون دارای ناسازگاری فرمتی/متنی هستند.",
        }

        # Data type issues: تعداد ستون‌های دارای مشکل نوع داده نسبت به کل
        n_dt_issue_cols = len(datatype_result)
        dt_ratio = n_dt_issue_cols / max(n_cols, 1)
        dt_penalty = min(self.WEIGHTS["data_type"], dt_ratio * self.WEIGHTS["data_type"] * 2)
        breakdown["data_type"] = {
            "penalty": round(dt_penalty, 2),
            "max_penalty": self.WEIGHTS["data_type"],
            "basis": f"{n_dt_issue_cols} از {n_cols} ستون مشکل نوع داده دارند.",
        }

        total_penalty = sum(b["penalty"] for b in breakdown.values())
        final_score = max(0.0, round(100 - total_penalty, 1))

        if final_score >= 90:
            verdict = "کیفیت عالی"
        elif final_score >= 75:
            verdict = "کیفیت خوب، نیاز به بررسی جزئی"
        elif final_score >= 55:
            verdict = "کیفیت متوسط، نیازمند Cleaning"
        else:
            verdict = "کیفیت پایین، نیازمند بازبینی جدی"

        return {
            "score": final_score,
            "verdict": verdict,
            "breakdown": breakdown,
            "disclaimer": (
                "این امتیاز صرفاً یک شاخص راهنما بر اساس قواعد وزن‌دهی‌شده‌ی از پیش تعریف‌شده است، "
                "نه یک معیار علمی قطعی. اهمیت واقعی هر مشکل به زمینه (Context) کسب‌وکار و هدف تحلیل بستگی دارد."
            ),
        }


# ==========================================================================
# کلاس اصلی هماهنگ‌کننده
# ==========================================================================

class DataQualityAnalyzer:
    """
    نقطه ورود اصلی ابزار. مسئول بارگذاری Dataset، اجرای تمام لایه‌های تحلیل،
    نمایش گزارش در ترمینال (Rich) و تولید گزارش HTML.

    مثال استفاده:
        analyzer = DataQualityAnalyzer("my_data.csv")
        analyzer.run()                          # پروفایلینگ + Detection + نمایش + HTML
        analyzer.export_html("report.html")     # اگر مسیر دلخواه دیگری بخواهید

    Cleaning تنها با فراخوانی صریح متدهای clean_* اجرا می‌شود و بخشی از run() نیست.
    """

    def __init__(self, file_path: Union[str, Path], sheet_name: Optional[Union[str, int]] = None,
                 config: Optional[dict] = None):
        self.file_path = str(file_path)
        self.sheet_name = sheet_name
        self.config = {**DEFAULT_CONFIG, **(config or {})}

        self.console = Console() if RICH_AVAILABLE else None

        self.load_result: Optional[LoadResult] = None
        self.df: Optional[pd.DataFrame] = None
        self.profile: Optional[dict] = None
        self.missing_result: Optional[dict] = None
        self.duplicate_result: Optional[dict] = None
        self.outlier_result: Optional[dict] = None
        self.invalid_result: Optional[dict] = None
        self.inconsistency_result: Optional[dict] = None
        self.datatype_result: Optional[dict] = None
        self.quality_score: Optional[dict] = None
        self.analysis_timestamp: Optional[str] = None
        self._analysis_complete = False

    # ---------------------------------------------------------------- #
    # اجرای کامل
    # ---------------------------------------------------------------- #

    def run(self, show_report: bool = True, export_html: bool = True,
            html_output_path: Optional[str] = None) -> bool:
        """
        اجرای کامل فرآیند: بارگذاری -> پروفایلینگ -> Detection همه‌جانبه -> نمایش -> خروجی HTML.
        در صورت شکست در هر مرحله، پیام خطای قابل‌فهم چاپ می‌شود و False برگردانده می‌شود؛
        اجرای برنامه هرگز به‌طور ناگهانی (exception خام) متوقف نمی‌شود.
        """
        try:
            if not self._load():
                return False

            self._print_status("در حال پروفایلینگ Dataset...")
            self.profile = DatasetProfiler(self.df, self.config).profile()

            self._print_status("در حال شناسایی Missing Values...")
            self.missing_result = MissingValueDetector(self.config).detect(self.df)

            self._print_status("در حال شناسایی Duplicate ها...")
            self.duplicate_result = DuplicateDetector(self.config).detect(
                self.df, identifier_columns=self.profile["identifier_columns"]
            )

            self._print_status("در حال شناسایی Outlier ها...")
            self.outlier_result = OutlierDetector(self.config).detect(self.df, self.profile["column_profiles"])

            self._print_status("در حال شناسایی Invalid Values...")
            self.invalid_result = InvalidValueDetector(self.config).detect(self.df, self.profile["column_profiles"])

            self._print_status("در حال شناسایی Inconsistency ها...")
            self.inconsistency_result = InconsistencyDetector(self.config).detect(self.df, self.profile["column_profiles"])

            self._print_status("در حال بررسی مشکلات Data Type...")
            self.datatype_result = DataTypeIssueDetector(self.config).detect(self.df, self.profile["column_profiles"])

            self.quality_score = QualityScoreCalculator().calculate(
                n_rows=self.profile["n_rows"], n_cols=self.profile["n_cols"],
                missing_result=self.missing_result, duplicate_result=self.duplicate_result,
                outlier_result=self.outlier_result, invalid_result=self.invalid_result,
                inconsistency_result=self.inconsistency_result, datatype_result=self.datatype_result,
            )

            self.analysis_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._analysis_complete = True

            if show_report:
                self.print_terminal_report()

            if export_html:
                out_path = html_output_path or self._default_html_path()
                self.export_html(out_path)

            return True

        except Exception as exc:  # noqa: BLE001
            self._print_error(f"خطای غیرمنتظره در اجرای تحلیل: {exc}\n{traceback.format_exc(limit=3)}")
            return False

    def _load(self) -> bool:
        self._print_status(f"در حال بارگذاری فایل: {self.file_path}")
        self.load_result = DatasetLoader.load(self.file_path, self.sheet_name)
        if not self.load_result.success:
            self._print_error(self.load_result.error_message)
            return False

        self.df = self.load_result.dataframe

        if self.df is None or self.df.shape[0] == 0:
            self._print_error("فایل بارگذاری شد اما هیچ ردیف داده‌ای ندارد.")
            return False
        if self.df.shape[1] == 0:
            self._print_error("فایل بارگذاری شد اما هیچ ستونی ندارد.")
            return False

        for w in self.load_result.load_warnings:
            self._print_warning(w)

        return True

    def _default_html_path(self) -> str:
        """
        مسیر پیش‌فرض خروجی HTML: کنار خودِ فایل Dataset ورودی ذخیره می‌شود
        (همان پوشه‌ای که فایل CSV/Excel در آن قرار دارد)، مگر اینکه صراحتاً
        مسیر دیگری به run()/export_html() داده شود.
        """
        input_path = Path(self.file_path).resolve()
        target_dir = input_path.parent
        stem = input_path.stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data_quality_report_{stem}_{timestamp}.html"
        return str(target_dir / filename)

    # ---------------------------------------------------------------- #
    # پیام‌های کمکی (با یا بدون Rich)
    # ---------------------------------------------------------------- #

    def _print_status(self, msg: str) -> None:
        if self.console:
            self.console.print(f"[cyan]›[/cyan] {msg}")
        else:
            print(f"› {msg}")

    def _print_warning(self, msg: str) -> None:
        if self.console:
            self.console.print(f"[yellow]⚠ هشدار:[/yellow] {msg}")
        else:
            print(f"⚠ هشدار: {msg}")

    def _print_error(self, msg: str) -> None:
        if self.console:
            self.console.print(f"[bold red]✗ خطا:[/bold red] {msg}")
        else:
            print(f"✗ خطا: {msg}")

    # ---------------------------------------------------------------- #
    # گزارش ترمینالی (Rich)
    # ---------------------------------------------------------------- #

    def print_terminal_report(self) -> None:
        if not self._analysis_complete:
            self._print_error("ابتدا باید run() اجرا شود.")
            return

        if not RICH_AVAILABLE or self.console is None:
            self._print_plain_report()
            return

        c = self.console
        c.print()
        c.print(Rule("[bold cyan]گزارش تحلیل کیفیت داده[/bold cyan]", style="cyan"))

        # --- پنل خلاصه کلی ---
        score = self.quality_score["score"]
        score_color = "green" if score >= 90 else "yellow" if score >= 75 else "red" if score < 55 else "yellow"
        summary_text = Text()
        summary_text.append(f"فایل: ", style="bold")
        summary_text.append(f"{self.file_path}\n")
        summary_text.append(f"زمان تحلیل: ", style="bold")
        summary_text.append(f"{self.analysis_timestamp}\n")
        summary_text.append(f"ابعاد: ", style="bold")
        summary_text.append(f"{self.profile['n_rows']:,} ردیف × {self.profile['n_cols']} ستون\n")
        summary_text.append(f"حجم حافظه: ", style="bold")
        summary_text.append(f"{self.profile['memory_human']}\n")
        summary_text.append(f"امتیاز کیفیت داده: ", style="bold")
        summary_text.append(f"{score}/100", style=f"bold {score_color}")
        summary_text.append(f"  ({self.quality_score['verdict']})", style=score_color)

        c.print(Panel(summary_text, title="خلاصه Dataset", border_style="cyan", box=box.ROUNDED))

        self._print_column_kind_table()
        self._print_missing_table()
        self._print_duplicate_panel()
        self._print_outlier_table()
        self._print_invalid_table()
        self._print_inconsistency_table()
        self._print_datatype_table()
        self._print_score_breakdown_table()

        c.print()
        c.print(Panel(
            "[bold]نکته:[/bold] در این مرحله فقط Profiling و Detection انجام شده است. "
            "هیچ تغییری در داده اعمال نشده. برای Cleaning، متدهای clean_* را صراحتاً فراخوانی کنید.",
            border_style="dim", box=box.ROUNDED,
        ))
        c.print(Rule(style="cyan"))

    def _print_column_kind_table(self) -> None:
        table = Table(title="نوع منطقی ستون‌ها", box=box.SIMPLE_HEAVY, show_lines=False)
        table.add_column("ستون", style="bold")
        table.add_column("نوع منطقی", style="magenta")
        table.add_column("dtype خام")
        table.add_column("دلیل تشخیص", overflow="fold")

        for col, prof in self.profile["column_profiles"].items():
            table.add_row(col, prof["kind"], self.profile["dtypes"].get(col, "-"), prof["reason"])
        self.console.print(table)

    def _print_missing_table(self) -> None:
        mv = self.missing_result
        if mv.get("error"):
            self._print_warning(mv["error"])
            return
        if mv["total_missing_cells"] == 0:
            self.console.print(Panel("هیچ Missing Value ای شناسایی نشد. ✅", border_style="green", box=box.ROUNDED))
            return

        table = Table(title=f"Missing Values (مجموع {mv['total_missing_cells']:,} سلول، "
                             f"{mv['overall_missing_pct']:.2f}% از کل)", box=box.SIMPLE_HEAVY)
        table.add_column("ستون", style="bold")
        table.add_column("تعداد", justify="right")
        table.add_column("درصد", justify="right")

        for col, info in mv["columns_sorted_by_missing"]:
            if info["n_missing"] == 0:
                continue
            table.add_row(col, f"{info['n_missing']:,}", f"{info['pct_missing']:.2f}%")
        self.console.print(table)

        for note in mv.get("missing_correlation_notes", []):
            self.console.print(f"[dim]  ↳ {note}[/dim]")

    def _print_duplicate_panel(self) -> None:
        dd = self.duplicate_result
        if dd.get("error"):
            self._print_warning(dd["error"])
            return
        if dd["n_duplicate_rows"] == 0 and not dd["identifier_duplicate_issues"]:
            self.console.print(Panel("هیچ ردیف یا شناسه تکراری‌ای شناسایی نشد. ✅", border_style="green", box=box.ROUNDED))
            return

        lines = [f"ردیف‌های کاملاً تکراری: {dd['n_duplicate_rows']:,} ({dd['pct_duplicate_rows']:.2f}%) "
                 f"در {dd['n_duplicate_groups']} گروه"]
        for issue in dd["identifier_duplicate_issues"]:
            lines.append(
                f"⚠ ستون شناسه «{issue['column']}» دارای {issue['n_duplicated_values']} مقدار تکراری "
                f"({issue['n_unique_duplicated']} مقدار یکتای تکرارشده) است — نمونه: {issue['sample_values']}"
            )
        self.console.print(Panel("\n".join(lines), title="Duplicate Rows / Values", border_style="yellow", box=box.ROUNDED))

    def _print_outlier_table(self) -> None:
        od = self.outlier_result
        if not od:
            self.console.print(Panel("ستون عددی برای تحلیل Outlier یافت نشد.", border_style="dim", box=box.ROUNDED))
            return

        table = Table(title="Outlier Detection (ستون‌های عددی)", box=box.SIMPLE_HEAVY, show_lines=True)
        table.add_column("ستون", style="bold")
        table.add_column("تعداد", justify="right")
        table.add_column("درصد", justify="right")
        table.add_column("روش‌های اعمال‌شده")
        table.add_column("دلیل انتخاب روش", overflow="fold")

        for col, info in od.items():
            if "error" in info:
                table.add_row(col, "-", "-", "خطا", info["error"])
                continue
            if info.get("n_outliers", 0) == 0 and "note" in info:
                table.add_row(col, "0", "0%", "-", info["note"])
                continue
            table.add_row(
                col, f"{info['n_outliers']:,}", f"{info['pct_outliers']:.2f}%",
                ", ".join(info["methods_used"]) or "-", info["selection_reasoning"],
            )
        self.console.print(table)

    def _print_invalid_table(self) -> None:
        iv = self.invalid_result
        if not iv:
            self.console.print(Panel("هیچ Invalid Value ای بر اساس قواعد شناخته‌شده یافت نشد. ✅",
                                      border_style="green", box=box.ROUNDED))
            return

        table = Table(title="Invalid Values (بر اساس قواعد منطقی)", box=box.SIMPLE_HEAVY, show_lines=True)
        table.add_column("ستون", style="bold")
        table.add_column("قاعده نقض‌شده", overflow="fold")
        table.add_column("تعداد", justify="right")
        table.add_column("نمونه مقادیر")

        for col, info in iv.items():
            if "error" in info:
                table.add_row(col, f"[red]خطا: {info['error']}[/red]", "-", "-")
                continue
            for rule in info["matched_rules"]:
                table.add_row(col, rule["rule"], f"{rule['n_invalid']:,}", str(rule["sample_values"][:5]))
        self.console.print(table)

    def _print_inconsistency_table(self) -> None:
        inc = self.inconsistency_result
        if not inc:
            self.console.print(Panel("ناسازگاری فرمتی/متنی قابل‌توجهی یافت نشد. ✅",
                                      border_style="green", box=box.ROUNDED))
            return

        table = Table(title="Inconsistencies", box=box.SIMPLE_HEAVY, show_lines=True)
        table.add_column("ستون", style="bold")
        table.add_column("نوع مشکل")
        table.add_column("توضیح", overflow="fold")

        for col, info in inc.items():
            if "error" in info:
                table.add_row(col, "خطا", info["error"])
                continue
            for issue in info["issues"]:
                table.add_row(col, issue["type"], issue["description"])
        self.console.print(table)

    def _print_datatype_table(self) -> None:
        dt = self.datatype_result
        if not dt:
            self.console.print(Panel("مشکل قابل‌توجهی در انطباق Data Type یافت نشد. ✅",
                                      border_style="green", box=box.ROUNDED))
            return

        table = Table(title="Data Type Issues", box=box.SIMPLE_HEAVY, show_lines=True)
        table.add_column("ستون", style="bold")
        table.add_column("dtype خام")
        table.add_column("توضیح", overflow="fold")

        for col, info in dt.items():
            if "error" in info:
                table.add_row(col, "-", info["error"])
                continue
            for issue in info["issues"]:
                table.add_row(col, info["raw_dtype"], issue["description"])
        self.console.print(table)

    def _print_score_breakdown_table(self) -> None:
        table = Table(title="تفکیک محاسبه Data Quality Score", box=box.SIMPLE_HEAVY)
        table.add_column("دسته")
        table.add_column("جریمه", justify="right")
        table.add_column("سقف جریمه", justify="right")
        table.add_column("مبنای محاسبه", overflow="fold")

        labels = {
            "missing": "Missing Values", "duplicates": "Duplicates", "outliers": "Outliers",
            "invalid": "Invalid Values", "inconsistency": "Inconsistencies", "data_type": "Data Type Issues",
        }
        for key, info in self.quality_score["breakdown"].items():
            table.add_row(labels.get(key, key), str(info["penalty"]), str(info["max_penalty"]), info["basis"])
        self.console.print(table)
        self.console.print(f"[dim]{self.quality_score['disclaimer']}[/dim]")

    def _print_plain_report(self) -> None:
        """نسخه ساده گزارش برای زمانی که rich در دسترس نیست."""
        print("\n" + "=" * 70)
        print("گزارش تحلیل کیفیت داده")
        print("=" * 70)
        print(f"فایل: {self.file_path}")
        print(f"ابعاد: {self.profile['n_rows']} ردیف × {self.profile['n_cols']} ستون")
        print(f"امتیاز کیفیت: {self.quality_score['score']}/100 ({self.quality_score['verdict']})")
        print(f"Missing cells: {self.missing_result.get('total_missing_cells', 'N/A')}")
        print(f"Duplicate rows: {self.duplicate_result.get('n_duplicate_rows', 'N/A')}")
        print("=" * 70 + "\n")

    # ---------------------------------------------------------------- #
    # خروجی HTML
    # ---------------------------------------------------------------- #

    def export_html(self, output_path: Optional[str] = None) -> Optional[str]:
        """تولید گزارش HTML مستقل (بدون نیاز به اینترنت) و ذخیره در مسیر مشخص یا مسیر پیش‌فرض."""
        if not self._analysis_complete:
            self._print_error("ابتدا باید run() اجرا شود تا export_html در دسترس باشد.")
            return None

        try:
            target_path = os.path.abspath(output_path or self._default_html_path())
            os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)

            html_content = HtmlReportBuilder(self).build()
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            self._print_status(f"✅ گزارش HTML ذخیره شد در مسیر کامل زیر:\n  {target_path}")
            return target_path
        except Exception as exc:  # noqa: BLE001
            self._print_error(f"خطا در تولید/ذخیره گزارش HTML: {exc}")
            return None


# ==========================================================================
# ساخت گزارش HTML مستقل
# ==========================================================================

def _esc(value: Any) -> str:
    """Escape ساده برای جلوگیری از شکستن HTML توسط مقادیر داده."""
    if value is None:
        return ""
    text = str(value)
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                .replace('"', "&quot;").replace("'", "&#39;"))


def _ltr(value: Any) -> str:
    """
    Escape + ایزوله‌سازی جهت برای مقادیر عددی/انگلیسی/تاریخ که داخل متن یا جدول RTL
    قرار می‌گیرند (مثل "502 ردیف"، "12.3%"، تاریخ‌ها، dtype ها). بدون این ایزوله‌سازی،
    الگوریتم Bidi مرورگر ترتیب نمایش این بخش‌ها را در بستر راست‌به‌چپ به‌هم می‌ریزد.
    """
    return f'<span dir="ltr" style="unicode-bidi:isolate;">{_esc(value)}</span>'


class HtmlReportBuilder:
    """
    ساخت گزارش HTML یک‌فایلی، مستقل و بدون نیاز به اینترنت.
    تمام CSS به‌صورت inline در تگ <style> قرار دارد و هیچ منبع خارجی فراخوانی نمی‌شود.
    نمودارها (در صورت وجود matplotlib) به‌صورت base64 داخل HTML جاسازی می‌شوند.
    """

    def __init__(self, analyzer: "DataQualityAnalyzer"):
        self.a = analyzer

    def build(self) -> str:
        a = self.a
        sections = [
            self._build_header(),
            self._build_summary_cards(),
            self._build_score_section(),
            self._build_column_overview_section(),
            self._build_missing_section(),
            self._build_duplicate_section(),
            self._build_outlier_section(),
            self._build_invalid_section(),
            self._build_inconsistency_section(),
            self._build_datatype_section(),
            self._build_footer_note(),
        ]
        body = "\n".join(sections)

        return f"""<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>گزارش کیفیت داده — {_esc(Path(a.file_path).name)}</title>
<style>
{self._build_css()}
</style>
</head>
<body>
<div class="container">
{body}
</div>
</body>
</html>"""

    # ------------------------------------------------------------------ #
    # CSS
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_css() -> str:
        return """
:root {
    --bg: #0f1420;
    --bg-card: #171d2e;
    --bg-card-alt: #1c2338;
    --border: #2a3350;
    --text-primary: #e8ecf5;
    --text-secondary: #9aa5c0;
    --text-dim: #6b7591;
    --accent: #5b8cff;
    --accent-soft: rgba(91, 140, 255, 0.12);
    --green: #4ade80;
    --green-soft: rgba(74, 222, 128, 0.12);
    --yellow: #fbbf24;
    --yellow-soft: rgba(251, 191, 36, 0.12);
    --red: #f87171;
    --red-soft: rgba(248, 113, 113, 0.12);
    --radius: 14px;
    font-family: 'Segoe UI', Tahoma, 'Vazirmatn', 'IRANSans', Arial, sans-serif;
}
* { box-sizing: border-box; }
body {
    background: linear-gradient(180deg, #0b0f19 0%, #0f1420 100%);
    color: var(--text-primary);
    margin: 0;
    padding: 32px 16px 64px;
    line-height: 1.75;
}
.container { max-width: 1180px; margin: 0 auto; }

/* Header */
.report-header {
    background: linear-gradient(135deg, #1a2036 0%, #141a2c 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 32px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.report-header::before {
    content: "";
    position: absolute; inset: 0;
    background: radial-gradient(600px circle at 85% -10%, var(--accent-soft), transparent 60%);
    pointer-events: none;
}
.report-header h1 { margin: 0 0 6px; font-size: 26px; font-weight: 700; color: #fff; }
.report-header .subtitle { color: var(--text-secondary); font-size: 14px; margin: 0; }
.header-meta {
    display: flex; flex-wrap: wrap; gap: 22px; margin-top: 20px;
    padding-top: 18px; border-top: 1px solid var(--border);
}
.header-meta .meta-item { font-size: 13px; color: var(--text-secondary); }
.header-meta .meta-item b { color: var(--text-primary); font-weight: 600; }

/* Summary cards */
.cards-grid {
    display: flex; flex-wrap: wrap;
    gap: 14px; margin-bottom: 24px;
}
.card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 18px 20px;
    flex: 1 1 160px; min-width: 160px;
}
.card .card-label { font-size: 12.5px; color: var(--text-dim); margin-bottom: 8px; }
.card .card-value { font-size: 26px; font-weight: 700; color: var(--text-primary); }
.card .card-sub { font-size: 12px; color: var(--text-secondary); margin-top: 4px; }
.card.accent-green .card-value { color: var(--green); }
.card.accent-yellow .card-value { color: var(--yellow); }
.card.accent-red .card-value { color: var(--red); }
.card.accent-blue .card-value { color: var(--accent); }

/* Sections */
.section {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 26px 28px; margin-bottom: 20px;
}
.section h2 {
    font-size: 18px; margin: 0 0 4px; color: #fff; display: flex; align-items: center; gap: 10px;
}
.section h2 .icon-dot {
    width: 8px; height: 8px; border-radius: 50%; background: var(--accent); flex-shrink: 0;
}
.section .section-desc { font-size: 13px; color: var(--text-secondary); margin: 0 0 18px; }
.section .empty-state {
    background: var(--green-soft); border: 1px solid rgba(74,222,128,0.25);
    color: var(--green); padding: 14px 18px; border-radius: 10px; font-size: 14px;
}

/* Score */
.score-wrap { display: flex; align-items: center; gap: 28px; flex-wrap: wrap; }
.score-circle {
    width: 120px; height: 120px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center; flex-direction: column;
    flex: 0 0 120px; border: 6px solid var(--border);
}
.score-circle .score-num { font-size: 28px; font-weight: 800; line-height: 1.2; }
.score-circle .score-max { font-size: 11px; color: var(--text-dim); }
.score-text-block { flex: 1 1 320px; min-width: 240px; }
.score-verdict { font-size: 15px; font-weight: 600; margin-bottom: 6px; }
.score-disclaimer { font-size: 12px; color: var(--text-dim); }

/* Tables */
table { width: 100%; border-collapse: collapse; font-size: 13.5px; margin-top: 6px; }
thead th {
    text-align: right; background: var(--bg-card-alt); color: var(--text-secondary);
    font-weight: 600; padding: 10px 14px; border-bottom: 1px solid var(--border);
    white-space: nowrap; position: sticky; top: 0;
}
tbody td { padding: 10px 14px; border-bottom: 1px solid var(--border); color: var(--text-primary); vertical-align: top; }
tbody tr:hover { background: rgba(255,255,255,0.02); }
tbody tr:last-child td { border-bottom: none; }
.table-scroll { overflow-x: auto; border-radius: 10px; border: 1px solid var(--border); }
.mono { font-family: 'Consolas', 'Courier New', monospace; font-size: 12.5px; color: var(--text-secondary); }

/* Badges */
.badge {
    display: inline-block; padding: 2px 10px; border-radius: 999px;
    font-size: 11.5px; font-weight: 600; white-space: nowrap;
}
.badge-blue { background: var(--accent-soft); color: var(--accent); }
.badge-green { background: var(--green-soft); color: var(--green); }
.badge-yellow { background: var(--yellow-soft); color: var(--yellow); }
.badge-red { background: var(--red-soft); color: var(--red); }
.badge-gray { background: rgba(154,165,192,0.12); color: var(--text-secondary); }

/* Progress bar (for score breakdown / missing %) */
.bar-track {
    background: var(--bg-card-alt); border-radius: 999px; height: 8px; width: 100%;
    overflow: hidden; min-width: 80px;
}
.bar-fill { height: 100%; border-radius: 999px; }

/* Column kind grid */
.kind-legend { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 16px; }
.kind-legend .legend-chip {
    font-size: 12px; padding: 5px 12px; border-radius: 8px;
    background: var(--bg-card-alt); color: var(--text-secondary); border: 1px solid var(--border);
}
.kind-legend .legend-chip b { color: var(--text-primary); }

.note-box {
    background: var(--accent-soft); border: 1px solid rgba(91,140,255,0.25);
    border-radius: 10px; padding: 14px 18px; font-size: 13px; color: var(--text-secondary); margin-top: 14px;
}
.footer-note {
    text-align: center; color: var(--text-dim); font-size: 12px; margin-top: 28px; line-height: 2;
}
.chart-wrap { margin-top: 18px; text-align: center; }
.chart-wrap img { max-width: 100%; border-radius: 10px; border: 1px solid var(--border); }

.sub-issue { margin-bottom: 10px; padding-bottom: 10px; border-bottom: 1px dashed var(--border); }
.sub-issue:last-child { border-bottom: none; margin-bottom: 0; padding-bottom: 0; }
.sub-issue .sub-issue-title { font-weight: 600; font-size: 13.5px; color: var(--text-primary); }
.sub-issue .sub-issue-desc { font-size: 12.5px; color: var(--text-secondary); margin-top: 2px; }

@media (max-width: 640px) {
    .report-header { padding: 22px 18px; }
    .section { padding: 18px 16px; }
    .score-wrap { gap: 16px; }
}
"""

    # ------------------------------------------------------------------ #
    # بخش‌های HTML
    # ------------------------------------------------------------------ #

    def _build_header(self) -> str:
        a = self.a
        p = a.profile
        return f"""
<div class="report-header">
    <h1>📊 گزارش تحلیل کیفیت داده</h1>
    <p class="subtitle">تولید شده به‌صورت خودکار — Data Quality &amp; Profiling Report</p>
    <div class="header-meta">
        <div class="meta-item">فایل: <b>{_ltr(Path(a.file_path).name)}</b></div>
        <div class="meta-item">زمان تحلیل: <b>{_ltr(a.analysis_timestamp)}</b></div>
        <div class="meta-item">ابعاد: <b>{_ltr(f"{p['n_rows']:,}")} ردیف × <b>{_ltr(p['n_cols'])}</b> ستون</b></div>
        <div class="meta-item">حجم حافظه: <b>{_ltr(p['memory_human'])}</b></div>
        <div class="meta-item">فرمت منبع: <b>{_ltr(a.load_result.file_format or '-')}</b></div>
    </div>
</div>
"""

    def _build_summary_cards(self) -> str:
        a = self.a
        p = a.profile
        mv = a.missing_result
        dd = a.duplicate_result
        n_outlier_cols = sum(1 for v in a.outlier_result.values() if v.get("n_outliers", 0) > 0)
        n_invalid_total = sum(v.get("n_total_flagged", 0) for v in a.invalid_result.values() if "error" not in v)

        def card(label, value, sub, accent=""):
            return f"""<div class="card {accent}">
                <div class="card-label">{_esc(label)}</div>
                <div class="card-value">{_ltr(value)}</div>
                <div class="card-sub">{sub}</div>
            </div>"""

        missing_pct_str = f"{mv.get('overall_missing_pct', 0):.2f}%"
        dup_pct_str = f"{dd.get('pct_duplicate_rows', 0):.2f}%"

        cards = [
            card("تعداد ردیف", f"{p['n_rows']:,}", f'{_ltr(p["n_cols"])} ستون', "accent-blue"),
            card("سلول‌های گمشده", f"{mv.get('total_missing_cells', 0):,}",
                 f'{_ltr(missing_pct_str)} از کل',
                 "accent-red" if mv.get('overall_missing_pct', 0) > 5 else "accent-green"),
            card("ردیف‌های تکراری", f"{dd.get('n_duplicate_rows', 0):,}",
                 f'{_ltr(dup_pct_str)} از کل',
                 "accent-red" if dd.get('n_duplicate_rows', 0) > 0 else "accent-green"),
            card("ستون‌های دارای Outlier", f"{n_outlier_cols}",
                 f'از {_ltr(len(a.outlier_result))} ستون عددی',
                 "accent-yellow" if n_outlier_cols > 0 else "accent-green"),
            card("مقادیر نامعتبر", f"{n_invalid_total:,}", "بر اساس قواعد منطقی",
                 "accent-red" if n_invalid_total > 0 else "accent-green"),
            card("ستون‌های شناسه", f"{len(p['identifier_columns'])}", "از تحلیل آماری کنار گذاشته شدند", "accent-blue"),
        ]
        return f'<div class="cards-grid">{"".join(cards)}</div>'

    def _build_score_section(self) -> str:
        a = self.a
        q = a.quality_score
        score = q["score"]
        color = "var(--green)" if score >= 90 else "var(--yellow)" if score >= 55 else "var(--red)"

        rows = []
        labels = {
            "missing": "Missing Values", "duplicates": "Duplicate Rows/Values", "outliers": "Outliers",
            "invalid": "Invalid Values", "inconsistency": "Inconsistencies", "data_type": "Data Type Issues",
        }
        for key, info in q["breakdown"].items():
            pct_of_max = (info["penalty"] / info["max_penalty"] * 100) if info["max_penalty"] else 0
            bar_color = "var(--red)" if pct_of_max > 60 else "var(--yellow)" if pct_of_max > 20 else "var(--green)"
            rows.append(f"""
            <tr>
                <td>{_esc(labels.get(key, key))}</td>
                <td style="width:160px;">
                    <div class="bar-track"><div class="bar-fill" style="width:{min(pct_of_max,100):.0f}%; background:{bar_color};"></div></div>
                </td>
                <td class="mono">{_ltr(f"{info['penalty']} / {info['max_penalty']}")}</td>
                <td>{_esc(info['basis'])}</td>
            </tr>""")

        return f"""
<div class="section">
    <h2><span class="icon-dot"></span> امتیاز کلی کیفیت داده (Data Quality Score)</h2>
    <p class="section-desc">این امتیاز صرفاً یک شاخص راهنماست، نه معیار علمی قطعی — جزئیات محاسبه در جدول زیر شفاف است.</p>
    <div class="score-wrap">
        <div class="score-circle" style="border-color:{color};">
            <div class="score-num" style="color:{color};">{_ltr(score)}</div>
            <div class="score-max">از ۱۰۰</div>
        </div>
        <div class="score-text-block">
            <div class="score-verdict" style="color:{color};">{_esc(q['verdict'])}</div>
            <div class="score-disclaimer">{_esc(q['disclaimer'])}</div>
        </div>
    </div>
    <div class="table-scroll" style="margin-top:20px;">
        <table>
            <thead><tr><th>دسته مشکل</th><th>سهم از جریمه</th><th>جریمه/سقف</th><th>مبنای محاسبه</th></tr></thead>
            <tbody>{"".join(rows)}</tbody>
        </table>
    </div>
</div>
"""

    def _build_column_overview_section(self) -> str:
        a = self.a
        p = a.profile
        kind_labels = {
            ColumnKind.NUMERIC: ("عددی", "badge-blue"),
            ColumnKind.CATEGORICAL: ("دسته‌ای", "badge-green"),
            ColumnKind.BOOLEAN: ("بولین", "badge-green"),
            ColumnKind.DATETIME: ("تاریخ/زمان", "badge-blue"),
            ColumnKind.TEXT_FREEFORM: ("متن آزاد", "badge-gray"),
            ColumnKind.IDENTIFIER: ("شناسه (ID)", "badge-yellow"),
            ColumnKind.CONSTANT: ("مقدار ثابت", "badge-red"),
            ColumnKind.EMPTY: ("کاملاً خالی", "badge-red"),
        }

        legend_items = "".join(
            f'<div class="legend-chip"><b>{p["kind_counts"].get(k,0)}</b> {label}</div>'
            for k, (label, _) in kind_labels.items() if p["kind_counts"].get(k, 0) > 0
        )

        rows = []
        for col, prof in p["column_profiles"].items():
            label, badge_cls = kind_labels.get(prof["kind"], (prof["kind"], "badge-gray"))
            rows.append(f"""
            <tr>
                <td><b>{_ltr(col)}</b></td>
                <td><span class="badge {badge_cls}">{_esc(label)}</span></td>
                <td class="mono">{_ltr(p['dtypes'].get(col, '-'))}</td>
                <td>{_esc(prof['reason'])}</td>
            </tr>""")

        return f"""
<div class="section">
    <h2><span class="icon-dot"></span> نمای کلی ستون‌ها و نوع منطقی داده</h2>
    <p class="section-desc">نوع منطقی هر ستون فراتر از dtype خام pandas تشخیص داده شده تا تحلیل‌های بعدی (Outlier، Invalid و ...) روی نوع درست اجرا شوند.</p>
    <div class="kind-legend">{legend_items}</div>
    <div class="table-scroll">
        <table>
            <thead><tr><th>ستون</th><th>نوع منطقی</th><th>dtype خام</th><th>دلیل تشخیص</th></tr></thead>
            <tbody>{"".join(rows)}</tbody>
        </table>
    </div>
</div>
"""

    def _build_missing_section(self) -> str:
        a = self.a
        mv = a.missing_result

        if mv.get("error"):
            body = f'<div class="empty-state" style="background:var(--red-soft); color:var(--red); border-color:rgba(248,113,113,0.25);">{_esc(mv["error"])}</div>'
        elif mv.get("total_missing_cells", 0) == 0:
            body = '<div class="empty-state">✅ هیچ Missing Value ای در Dataset شناسایی نشد.</div>'
        else:
            rows = []
            for col, info in mv["columns_sorted_by_missing"]:
                if info["n_missing"] == 0:
                    continue
                pct = info["pct_missing"]
                bar_color = "var(--red)" if pct > 20 else "var(--yellow)" if pct > 5 else "var(--green)"
                rows.append(f"""
                <tr>
                    <td><b>{_ltr(col)}</b></td>
                    <td class="mono">{_ltr(f"{info['n_missing']:,}")}</td>
                    <td style="width:180px;">
                        <div style="display:flex; align-items:center; gap:10px;">
                            <div class="bar-track"><div class="bar-fill" style="width:{min(pct,100):.0f}%; background:{bar_color};"></div></div>
                            <span class="mono">{_ltr(f"{pct:.2f}%")}</span>
                        </div>
                    </td>
                </tr>""")
            notes_html = ""
            if mv.get("missing_correlation_notes"):
                notes = "".join(f"<li>{_esc(n)}</li>" for n in mv["missing_correlation_notes"])
                notes_html = f'<div class="note-box"><b>الگوهای هم‌رخدادی:</b><ul>{notes}</ul></div>'

            body = f"""
            <div class="table-scroll">
                <table>
                    <thead><tr><th>ستون</th><th>تعداد گمشده</th><th>درصد</th></tr></thead>
                    <tbody>{"".join(rows)}</tbody>
                </table>
            </div>
            <p class="section-desc" style="margin-top:14px;">
                {_ltr(f"{mv['rows_with_any_missing']:,}")} ردیف حداقل یک مقدار گمشده دارند
                (از {_ltr(f"{a.profile['n_rows']:,}")} ردیف کل)،
                و {_ltr(f"{mv['rows_fully_missing']:,}")} ردیف کاملاً خالی هستند.
            </p>
            {notes_html}
            """

        return f"""
<div class="section">
    <h2><span class="icon-dot"></span> Missing Values</h2>
    <p class="section-desc">شامل NaN واقعی و رشته‌های معادل آن (مثل "N/A"، "-"، "null" و ...).</p>
    {body}
</div>
"""

    def _build_duplicate_section(self) -> str:
        a = self.a
        dd = a.duplicate_result

        if dd.get("error"):
            body = f'<div class="empty-state" style="background:var(--red-soft); color:var(--red); border-color:rgba(248,113,113,0.25);">{_esc(dd["error"])}</div>'
        elif dd.get("n_duplicate_rows", 0) == 0 and not dd.get("identifier_duplicate_issues"):
            body = '<div class="empty-state">✅ هیچ ردیف یا شناسه تکراری‌ای شناسایی نشد.</div>'
        else:
            parts = [f"""
            <div class="card" style="margin-bottom:16px;">
                <div class="card-label">ردیف‌های کاملاً تکراری</div>
                <div class="card-value">{_ltr(f"{dd['n_duplicate_rows']:,}")} <span style="font-size:14px; color:var(--text-dim);">{_ltr(f"({dd['pct_duplicate_rows']:.2f}%)")}</span></div>
                <div class="card-sub">در {_ltr(dd['n_duplicate_groups'])} گروه مجزا</div>
            </div>"""]

            if dd.get("identifier_duplicate_issues"):
                id_rows = []
                for issue in dd["identifier_duplicate_issues"]:
                    sample = ", ".join(_esc(str(v)) for v in issue["sample_values"])
                    id_rows.append(f"""
                    <tr>
                        <td><b>{_ltr(issue['column'])}</b></td>
                        <td class="mono">{_ltr(f"{issue['n_duplicated_values']:,}")}</td>
                        <td class="mono">{_ltr(f"{issue['n_unique_duplicated']:,}")}</td>
                        <td class="mono">{_ltr(sample)}</td>
                    </tr>""")
                parts.append(f"""
                <div class="note-box" style="background:var(--yellow-soft); border-color: rgba(251,191,36,0.25); margin-bottom:14px;">
                    ⚠ برخی ستون‌های شناسه (که باید یکتا باشند) دارای مقادیر تکراری هستند.
                </div>
                <div class="table-scroll">
                    <table>
                        <thead><tr><th>ستون شناسه</th><th>تعداد ردیف تکرارشده</th><th>مقادیر یکتای تکراری</th><th>نمونه مقادیر</th></tr></thead>
                        <tbody>{"".join(id_rows)}</tbody>
                    </table>
                </div>""")

            body = "".join(parts)

        return f"""
<div class="section">
    <h2><span class="icon-dot"></span> Duplicate Rows / Values</h2>
    <p class="section-desc">ردیف‌های کاملاً یکسان در تمام ستون‌ها، و همچنین تکرار در ستون‌هایی که انتظار یکتایی می‌رود (شناسه‌ها).</p>
    {body}
</div>
"""

    def _build_outlier_section(self) -> str:
        a = self.a
        od = a.outlier_result

        if not od:
            body = '<div class="empty-state" style="background:rgba(154,165,192,0.1); color:var(--text-secondary); border-color:var(--border);">ستون عددی‌ای برای تحلیل Outlier در این Dataset یافت نشد.</div>'
        else:
            blocks = []
            for col, info in od.items():
                if "error" in info:
                    blocks.append(f'<div class="sub-issue"><div class="sub-issue-title">{_ltr(col)}</div><div class="sub-issue-desc" style="color:var(--red);">{_esc(info["error"])}</div></div>')
                    continue
                if info.get("n_outliers", 0) == 0:
                    note = info.get("note", "Outlier ای شناسایی نشد.")
                    blocks.append(f"""
                    <div class="sub-issue">
                        <div class="sub-issue-title">{_ltr(col)} <span class="badge badge-green">{_ltr("0")} outlier</span></div>
                        <div class="sub-issue-desc">{_esc(note)}</div>
                    </div>""")
                    continue

                methods_badges = " ".join(f'<span class="badge badge-blue">{_ltr(m)}</span>' for m in info["methods_used"])
                sample_vals = ", ".join(f"{r['value']:.2f}" for r in info["outlier_records_sample"][:8])

                if "lower_bound" in info["iqr"]:
                    iqr_bounds_str = f"{info['iqr']['lower_bound']:.2f} تا {info['iqr']['upper_bound']:.2f}"
                else:
                    iqr_bounds_str = info["iqr"].get("note", "قابل محاسبه نبود")

                extra_count = f" (و {_ltr(info['n_outliers']-8)} مورد دیگر)" if info['n_outliers'] > 8 else ""

                blocks.append(f"""
                <div class="sub-issue">
                    <div class="sub-issue-title">{_ltr(col)}
                        <span class="badge badge-yellow">{_ltr(f"{info['n_outliers']:,} outlier ({info['pct_outliers']:.2f}%)")}</span>
                        {methods_badges}
                    </div>
                    <div class="sub-issue-desc">
                        <b>دلیل انتخاب روش:</b> {_esc(info['selection_reasoning'])}<br>
                        <b>بازه نرمال (IQR):</b> {_ltr(iqr_bounds_str)}<br>
                        <b>آمار توصیفی:</b> {_ltr(f"میانگین={info['mean']:.2f}, میانه={info['median']:.2f}, انحراف‌معیار={info['std']:.2f}, دامنه=[{info['min_value']:.2f}, {info['max_value']:.2f}]")}<br>
                        <b>نمونه مقادیر پرت:</b> <span class="mono">{_ltr(sample_vals)}</span>
                        {extra_count}
                    </div>
                </div>""")
            body = "".join(blocks)

        return f"""
<div class="section">
    <h2><span class="icon-dot"></span> Outlier Detection</h2>
    <p class="section-desc">
        فقط روی ستون‌های عددی اجرا می‌شود. روش‌ها بر اساس حجم نمونه و چولگی (Skewness) توزیع هر ستون به‌صورت خودکار انتخاب می‌شوند:
        IQR (مقاوم، همیشه محاسبه می‌شود)، Z-Score (فقط برای توزیع‌های نزدیک به نرمال با نمونه کافی)،
        و Modified Z-Score مبتنی بر MAD (مقاوم در برابر چولگی و outlier شدید).
    </p>
    {body}
</div>
"""

    def _build_invalid_section(self) -> str:
        a = self.a
        iv = a.invalid_result

        if not iv:
            body = '<div class="empty-state">✅ بر اساس قواعد شناخته‌شده، مقدار غیرمنطقی/غیرممکنی یافت نشد.</div>'
        else:
            rows = []
            for col, info in iv.items():
                if "error" in info:
                    rows.append(f'<tr><td><b>{_ltr(col)}</b></td><td colspan="3" style="color:var(--red);">{_esc(info["error"])}</td></tr>')
                    continue
                for rule in info["matched_rules"]:
                    sample = ", ".join(_esc(str(v)) for v in rule["sample_values"][:6])
                    rows.append(f"""
                    <tr>
                        <td><b>{_ltr(col)}</b></td>
                        <td>{_esc(rule['rule'])}</td>
                        <td class="mono">{_ltr(f"{rule['n_invalid']:,} ({rule['pct_invalid']:.2f}%)")}</td>
                        <td class="mono">{_ltr(sample)}</td>
                    </tr>""")
            body = f"""
            <div class="table-scroll">
                <table>
                    <thead><tr><th>ستون</th><th>قاعده نقض‌شده</th><th>تعداد</th><th>نمونه مقادیر</th></tr></thead>
                    <tbody>{"".join(rows)}</tbody>
                </table>
            </div>
            """

        return f"""
<div class="section">
    <h2><span class="icon-dot"></span> Invalid Values</h2>
    <p class="section-desc">مقادیری که از نظر منطقی/دامنه‌ای غیرممکن‌اند (مثل سن منفی یا درصد خارج از بازه ۰ تا ۱۰۰)، فارغ از اینکه از نظر آماری Outlier باشند یا نه.</p>
    {body}
</div>
"""

    def _build_inconsistency_section(self) -> str:
        a = self.a
        inc = a.inconsistency_result

        if not inc:
            body = '<div class="empty-state">✅ ناسازگاری فرمتی/متنی قابل‌توجهی شناسایی نشد.</div>'
        else:
            blocks = []
            for col, info in inc.items():
                if "error" in info:
                    blocks.append(f'<div class="sub-issue"><div class="sub-issue-title">{_ltr(col)}</div><div class="sub-issue-desc" style="color:var(--red);">{_esc(info["error"])}</div></div>')
                    continue
                for issue in info["issues"]:
                    extra = ""
                    if issue["type"] == "case_mismatch":
                        examples = " | ".join(
                            f"{_esc(ex['normalized'])} = [{', '.join(_esc(v) for v in ex['variants'])}]"
                            for ex in issue["examples"]
                        )
                        extra = f'<div class="mono" style="margin-top:6px;" dir="ltr">{examples}</div>'
                    elif issue["type"] == "leading_trailing_whitespace":
                        examples = ", ".join(_esc(e) for e in issue["examples"])
                        extra = f'<div class="mono" style="margin-top:6px;">نمونه: <span dir="ltr" style="unicode-bidi:isolate;">{examples}</span></div>'
                    elif issue["type"] == "inconsistent_boolean_representation":
                        true_vals = ", ".join(_esc(v) for v in issue["true_variants"])
                        false_vals = ", ".join(_esc(v) for v in issue["false_variants"])
                        extra = (f'<div class="mono" style="margin-top:6px;">'
                                 f'مقادیر True: <span dir="ltr" style="unicode-bidi:isolate;">{true_vals}</span> | '
                                 f'مقادیر False: <span dir="ltr" style="unicode-bidi:isolate;">{false_vals}</span></div>')
                    blocks.append(f"""
                    <div class="sub-issue">
                        <div class="sub-issue-title">{_ltr(col)} <span class="badge badge-yellow">{_ltr(issue['type'])}</span></div>
                        <div class="sub-issue-desc">{_esc(issue['description'])}{extra}</div>
                    </div>""")
            body = "".join(blocks)

        return f"""
<div class="section">
    <h2><span class="icon-dot"></span> Inconsistencies</h2>
    <p class="section-desc">ناسازگاری در نمایش مقادیر معنایی یکسان: تفاوت حروف بزرگ/کوچک، فاصله‌های اضافی، یا نمایش‌های ناهمگون boolean.</p>
    {body}
</div>
"""

    def _build_datatype_section(self) -> str:
        a = self.a
        dt = a.datatype_result

        if not dt:
            body = '<div class="empty-state">✅ مشکل قابل‌توجهی در انطباق نوع داده (Data Type) یافت نشد.</div>'
        else:
            blocks = []
            for col, info in dt.items():
                if "error" in info:
                    blocks.append(f'<div class="sub-issue"><div class="sub-issue-title">{_ltr(col)}</div><div class="sub-issue-desc" style="color:var(--red);">{_esc(info["error"])}</div></div>')
                    continue
                for issue in info["issues"]:
                    blocks.append(f"""
                    <div class="sub-issue">
                        <div class="sub-issue-title">{_ltr(col)} <span class="badge badge-blue mono">{_ltr(info['raw_dtype'])}</span></div>
                        <div class="sub-issue-desc">{_esc(issue['description'])}</div>
                    </div>""")
            body = "".join(blocks)

        return f"""
<div class="section">
    <h2><span class="icon-dot"></span> Data Type Issues</h2>
    <p class="section-desc">ستون‌هایی که نوع ذخیره‌سازی خام آن‌ها (dtype) با محتوای واقعی داده همخوانی ندارد.</p>
    {body}
</div>
"""

    def _build_footer_note(self) -> str:
        return """
<div class="footer-note">
    این گزارش صرفاً شامل Profiling و Detection است — هیچ تغییری در داده اصلی اعمال نشده است.<br>
    برای Cleaning و اصلاح داده، لازم است این عملیات به‌صورت جداگانه و صریح درخواست شود.<br>
    تولید شده با Data Quality Analyzer — Python / Pandas / Rich
</div>
"""
