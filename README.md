# تحلیل اثربخشی برنامه‌های آموزشی (L&D ROI) — Meridian Group

*This document is bilingual; the Persian version appears first, followed by this English version.*

پروژه‌ی تحلیل داده / تحلیل کسب‌وکار / تحلیل منابع انسانی برای پاسخ به یک سوال واحد:

> **آیا ۲.۷۳ میلیون دلار هزینه‌ی آموزشی Meridian Group در بازه‌ی ۲۰۲۳ تا ۲۰۲۵، واقعاً روی عملکرد، ترفیع، و ماندگاری کارکنان اثر داشته است؟**

---

## سناریو

Meridian Group هلدینگی با ۳,۵۰۰ کارمند در ۴ دپارتمان (Sales, Engineering, Customer Support, Operations) و ۶ کشور است. بورد پیش از تایید بودجه‌ی L&D سال ۲۰۲۶، خواستار مدرکی داده‌محور است — نه صرفاً رضایت شرکت‌کنندگان از دوره‌ها (که گزارش مشابه دو سال پیش رد شده بود).

شرح کامل سناریو و فرآیند کشف مسئله در دو فایل زیر آمده:
- [`سناریو.pdf`](./سناریو.pdf)
- [`کشف و درک مسئله.pdf`](./کشف%20و%20درک%20مسئله.pdf)

---

## ساختار پروژه

```
Meridian Group/
├── classes/                          کلاس‌های پایتون قابل‌استفاده‌ی مجدد
│   ├── data_quality_analyzer.py      تشخیص کامل کیفیت داده (missing/duplicate/outlier/invalid/formatting)
│   └── data_info_display.py         نمایش خلاصه‌ی سریع هر دیتاست
│
├── datasets/                         داده‌ی خام + گزارش‌های کیفیت هر جدول
│   ├── employees.csv
│   ├── hr_events.csv
│   ├── performance_records.csv
│   ├── training_enrollments.csv
│   ├── training_programs.csv
│   ├── data_quality_report_*.html    گزارش کیفیت هر ۵ جدول (خروجی classes/)
│   └── ستون ها و معانی هریک.html      دیکشنری داده
│
├── stakeholders/                      نقشه‌ی ذی‌نفعان (دوزبانه، فارسی/انگلیسی)
│
├── combined_employee_dataset.parquet  دیتاست نهایی ترکیب‌شده - یک سطر برای هر کارمند
│
├── Understanding and Cleaning.ipynb   کشف مسئله، بررسی کیفیت هر جدول، تصمیمات پاکسازی
├── EDA.ipynb                          تحلیل اکتشافی + تحلیل عمیق (stratified) + ترکیب نهایی
│
├── Report(for any stakeholder).html   گزارش یافته‌ها، جدا برای هر ۴ ذی‌نفع (CFO/L&D/دپارتمان/HRBP)
└── Insight.html                       سند Business Analysis: توصیه‌های عملی + ریسک عدم‌اجرا
```

---

## ذی‌نفعان

| ذی‌نفع | دغدغه اصلی |
|---|---|
| CFO / CEO | توجیه مالی هزینه‌ی ۲.۸ میلیون دلاری |
| مدیر L&D | کدام‌یک از ۳۱ دوره‌ی موجود ادامه یابد یا حذف شود |
| مدیران دپارتمان (Sales/Engineering/CS/Ops) | آیا تیم‌شان از آموزش سود برده‌اند |
| HRBP | رابطه‌ی آموزش با نرخ خروج (Exit) |

جزئیات کامل در پوشه‌ی `stakeholders/`.

---

## مسیر کار (به ترتیب اجرا/مطالعه)

1. **`سناریو.pdf`** و **`کشف و درک مسئله.pdf`** — درک زمینه و هدف پروژه
2. **`datasets/`** — بررسی داده‌ی خام و گزارش کیفیت هر جدول (تولید‌شده با `classes/data_quality_analyzer.py`)
3. **`Understanding and Cleaning.ipynb`** — تصمیمات پاکسازی مستند و اجرا شده
4. **`EDA.ipynb`** — ترکیب ۵ جدول به `combined_employee_dataset.parquet`، تحلیل اکتشافی، و تحلیل عمیق با کنترل `job_level` (برای رفع نگرانی selection bias)
5. **`Report(for any stakeholder).html`** — نتیجه‌ی نهایی برای هر ذی‌نفع
6. **`Insight.html`** — توصیه‌های عملی و قدم بعدی

---

## یافته‌های کلیدی

- کارمندان آموزش‌دیده نرخ ترفیع **~۱.۷ برابر** بیشتر و نرخ خروج **~۲ برابر** کمتر داشته‌اند.
- این رابطه با کنترل سطح شغلی (برای رفع تورش انتخاب/selection bias) تا حد زیادی پابرجا مانده است.
- برخلاف فرض اولیه، دوره‌های **Compliance** (نه Leadership) بیشترین فاصله در نرخ ترفیع را نشان می‌دهند.
- این یک مطالعه‌ی *observational* است، نه آزمایش تصادفی — نتایج همبستگی (correlation) هستند، نه اثبات قطعی علیت.

جزئیات کامل، اعداد دقیق، و محدودیت‌های هر یافته در `Report(for any stakeholder).html` و `Insight.html`.

---

## ابزارها و روش‌ها

زبان و کتابخانه‌ها: `Python (pandas, NumPy, SciPy)` · `Jupyter` — به‌همراه تشخیص کیفیت داده به‌صورت کلاس قابل‌استفاده‌ی مجدد · ترکیب چندجدولی با پرهیز از fan-out join · تحلیل stratified با Welch's t-test · تولید گزارش HTML مستقل برای هر ذی‌نفع

---

## محدودیت‌ها

۱. مطالعه observational است؛ علیت قطعی قابل اثبات نیست.
۲. در برخی سطوح شغلی (مثلاً Director)، حجم گروه کنترل کوچک بود و آزمون آماری قدرت کافی نداشت.
۳. رابطه‌ی آموزش با نرخ خروج نیز correlation است — ممکن است افراد باانگیزه‌تر هم بیشتر آموزش دیده باشند هم بیشتر مانده باشند.

---
---

# L&D Effectiveness Analysis (L&D ROI) — Meridian Group

A Data Analytics / Business Analysis / HR Analytics project built to answer a single question:

> **Did Meridian Group's $2.73 million training investment between 2023 and 2025 actually affect employee performance, promotion, and retention?**

---

## Scenario

Meridian Group is a 3,500-employee holding company with 4 business units (Sales, Engineering, Customer Support, Operations) across 6 countries. Before approving the 2026 L&D budget, the board is demanding data-driven evidence — not just participant satisfaction scores (a similar report was rejected two years ago for exactly this reason).

The full scenario and problem-discovery process are documented in:
- [`سناریو.pdf`](./سناریو.pdf) *(Scenario)*
- [`کشف و درک مسئله.pdf`](./کشف%20و%20درک%20مسئله.pdf) *(Problem Discovery)*

---

## Project Structure

```
Meridian Group/
├── classes/                          Reusable Python classes
│   ├── data_quality_analyzer.py      Full data-quality inspection (missing/duplicate/outlier/invalid/formatting)
│   └── data_info_display.py         Quick summary display for any dataset
│
├── datasets/                         Raw data + per-table quality reports
│   ├── employees.csv
│   ├── hr_events.csv
│   ├── performance_records.csv
│   ├── training_enrollments.csv
│   ├── training_programs.csv
│   ├── data_quality_report_*.html    Quality report for each of the 5 tables (output of classes/)
│   └── ستون ها و معانی هریک.html      Data dictionary
│
├── stakeholders/                      Stakeholder map (bilingual, Persian/English)
│
├── combined_employee_dataset.parquet  Final merged dataset - one row per employee
│
├── Understanding and Cleaning.ipynb   Problem discovery, per-table quality review, cleaning decisions
├── EDA.ipynb                          Exploratory analysis + stratified deep-dive + final merge
│
├── Report(for any stakeholder).html   Findings report, split by each of the 4 stakeholders (CFO/L&D/Depts/HRBP)
└── Insight.html                       Business Analysis document: actionable recommendations + risk of inaction
```

---

## Stakeholders

| Stakeholder | Primary Concern |
|---|---|
| CFO / CEO | Financial ROI and justification for the $2.8 million cost |
| L&D Manager | Which of the 31 available courses to continue or discontinue |
| Department Managers (Sales/Engineering/CS/Ops) | Whether their team benefited from the training |
| HRBP | The relationship between training and exit rate |

Full details in the `stakeholders/` folder.

---

## Workflow (in order of execution/review)

1. **`سناریو.pdf`** and **`کشف و درک مسئله.pdf`** — understand the project's context and goal
2. **`datasets/`** — review raw data and each table's quality report (produced with `classes/data_quality_analyzer.py`)
3. **`Understanding and Cleaning.ipynb`** — documented and executed cleaning decisions
4. **`EDA.ipynb`** — merges the 5 tables into `combined_employee_dataset.parquet`, exploratory analysis, and stratified deep-dive controlling for `job_level` (to address the selection-bias concern)
5. **`Report(for any stakeholder).html`** — final findings for each stakeholder
6. **`Insight.html`** — actionable recommendations and next steps

---

## Key Findings

- Trained employees had a **~1.7x** higher promotion rate and a **~2x** lower exit rate.
- This relationship largely held up after controlling for job level (to address selection bias).
- Contrary to the initial hypothesis, **Compliance** courses (not Leadership) show the widest promotion-rate gap.
- This is an *observational* study, not a randomized experiment — the results are correlational, not definitive proof of causation.

Full details, exact figures, and per-finding limitations are in `Report(for any stakeholder).html` and `Insight.html`.

---

## Tools & Methods

Language and libraries: `Python (pandas, NumPy, SciPy)` · `Jupyter` — along with data-quality inspection as a reusable class · multi-table merging while avoiding fan-out joins · stratified analysis with Welch's t-test · standalone HTML report generation for each stakeholder

---

## Limitations

1. This is an observational study; definitive causation cannot be established.
2. For some job levels (e.g., Director), the control-group sample size was small, limiting the statistical power of the test.
3. The relationship between training and exit rate is also correlational — it's possible that more motivated employees are both more likely to train and more likely to stay.

