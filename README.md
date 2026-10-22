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
│   ├── 05_peak_submission.py       # Final Rank-Averaged Ensemble achieving Peak Global Rank #1 (0.80372 AUC)
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

## 1. Credit Default Risk Competition (Global Rank #1 Peak)

### Performance Highlights
* **Peak Global Leaderboard Rank:** **#1**
* **Best Test ROC-AUC:** **0.80372**
* **Validation Strategy:** Leak-free 5-Fold Stratified Cross-Validation with out-of-fold probability calibration.

### Key Engineering & Modeling Components
1. **Domain Feature Engineering (40+ features):**
   - **Financial Leverage Ratios:** `Credit/Income`, `Annuity/Income`, `Goods_Price/Credit`.
   - **Loan-Term & Implied Interest Rate:** Clustered `(Credit, Annuity)` to extract standard loan durations and derived the hidden implied interest rate charged by the lender using numerical bisection.
   - **External Bureau Aggregations:** Non-linear interactions, polynomial products, and dispersion statistics across `EXT_SOURCE_1, 2, 3`.
   - **Repeat-Applicant Fingerprinting:** Computed historical application default tendencies strictly within training folds using Bayesian-smoothed target encoding.

2. **Model Diversity & Rank-Averaged Ensembling:**
   - **LightGBM:** Leaf-wise split growth for deep feature interactions.
   - **CatBoost:** Symmetric/oblivious decision trees to prevent tabular overfitting.
   - **XGBoost:** Depth-wise histogram-based tree growth with L1/L2 shrinkage.
   - **Ensemble Method:** Multi-seed rank averaging (`scipy.stats.rankdata`) to directly optimize ranking order and eliminate probability calibration divergence.

---

## 2. Telecom Customer Churn Capstone & Business Proposal

An end-to-end churn prediction framework and quantitative retention strategy for a subscription enterprise:
* **Longitudinal Behavioral Modeling:** Tracks customer velocity across data consumption, payment regularity, and service interaction history.
* **Risk Stratification:** Segments users into Low ($P < 0.2$), Medium ($0.2 \le P < 0.6$), and High ($P \ge 0.6$) churn risk tiers.
* **Executive Decision Proposal:** Formulates an ROI optimization framework ensuring customer retention interventions are only deployed when expected lifetime value saved exceeds total campaign cost.

---

## Author
* **Jayanth Malireddi** — Indian Institute of Technology (IIT) Indore
* GitHub: [@jay-git00](https://github.com/jay-git00)
