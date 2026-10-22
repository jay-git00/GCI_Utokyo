# Global Consumer Intelligence (GCI) Data Science Program
### Matsuo-Iwasawa Laboratory — The University of Tokyo

This repository contains the end-to-end data pipelines, predictive models, and business proposals developed as part of the **Global Consumer Intelligence (GCI) Data Science Program** organized by the Matsuo-Iwasawa Laboratory at The University of Tokyo.

---

## Repository Structure

```
├── competition_data/              # In-Program ML Competition: Credit Default Risk Modeling
│   ├── 01_data_audit.py           # Exploratory data audit & anomaly detection (e.g. DAYS_EMPLOYED)
│   ├── 02_full_pipeline.py         # Baseline LightGBM pipeline with stratified cross-validation
│   ├── 03_nuclear_pipeline.py      # Expanded feature engineering pipeline (ratios, EXT_SOURCE combos)
│   ├── 04_champion_pipeline.py     # Tri-Model ensemble (LightGBM + CatBoost + XGBoost)
│   ├── 05_peak_submission.py       # Multi-seed ensembling and out-of-fold blending
│   ├── boost_rank.py              # Calibration and out-of-fold blending optimizer
│   └── verify_submission.py       # Validation and submission shape verification script
│
├── final_assignment/              # Capstone Assignment: Telecom Churn Prediction & Business Proposal
│   ├── tutorial.ipynb             # Exploratory Data Analysis & longitudinal modeling notebook
│   ├── ENG_Company _A_ Dataset Overview.docx # Dataset documentation and column definitions
│   └── README.docx                # Rubric and deliverable requirements
│
├── lecture_1_syllabus.md          # Program syllabus and curriculum overview
├── student_guide.md               # Student handbook and lecture schedule
└── .gitignore                     # Data and cache exclusion rules
```

---

## 1. Credit Default Risk Competition

### Objective & Architecture
Development of a robust credit risk scoring system on the Home Credit Default Risk dataset, focusing on rigorous feature engineering, data anomaly resolution, and ensemble modeling under class-imbalanced conditions (~8% default rate).

### Key Engineering & Modeling Components
1. **Data Cleaning & Anomaly Correction:**
   - Identified and corrected legacy placeholder anomalies (e.g., converting `DAYS_EMPLOYED = 365243` to `NaN` and creating an explicit anomaly flag to prevent skewed tree splits).
   - Handled categorical missing values and invalid placeholders (`XNA`).

2. **Domain Feature Engineering (40+ features):**
   - **Financial Leverage Ratios:** `Credit/Income` (debt-to-income multiplier), `Annuity/Income` (repayment burden), `Goods_Price/Credit` (over-financing indicator).
   - **Loan-Term & Implied Interest Rate:** Clustered `(Credit, Annuity)` to extract standard repayment durations and estimated borrower implied interest rates via root-finding on the annuity equation.
   - **External Bureau Aggregations:** Non-linear interactions, polynomial products, and dispersion statistics across `EXT_SOURCE_1, 2, 3`.
   - **Repeat-Applicant Fingerprinting:** Computed historical application tendencies strictly within training folds using Bayesian-smoothed target encoding to guarantee leak-free validation.

3. **Validation & Ensemble Strategy:**
   - **Validation:** 5-Fold Stratified Cross-Validation with strict fold isolation for all encoders and scalers.
   - **Model Diversity:** Evaluated LightGBM (leaf-wise growth for high-order interactions), CatBoost (symmetric trees robust to tabular noise), and XGBoost.
   - **Blending:** Multi-seed rank averaging (`scipy.stats.rankdata`) to optimize ranking order across heterogeneous model outputs.

---

## 2. Telecom Customer Churn Capstone & Business Proposal

An end-to-end churn prediction framework and quantitative retention strategy for a subscription enterprise:
* **Longitudinal Behavioral Modeling:** Tracks customer velocity across data consumption, payment regularity, and customer service interaction history.
* **Risk Stratification:** Segments users into Low ($P < 0.2$), Medium ($0.2 \le P < 0.6$), and High ($P \ge 0.6$) churn risk tiers.
* **Executive Decision Proposal:** Formulates an ROI optimization framework ensuring customer retention interventions are only deployed when expected lifetime value saved exceeds total campaign cost.

---

## Author
* **Jayanth Malireddi** — Indian Institute of Technology (IIT) Indore
* GitHub: [@jay-git00](https://github.com/jay-git00)
