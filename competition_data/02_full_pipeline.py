"""
=============================================================================
GCI World 2026 September - Home Credit Default Risk
Competition Pipeline: Top 1% Target
=============================================================================
Author: Antigravity AI (10+ yr professional data scientist mode)
Strategy:
  1. Elite feature engineering (60+ features from 33 raw features)
  2. 5-Fold Stratified LightGBM with calibrated params
  3. 5-Fold Stratified CatBoost
  4. Rank-averaged ensemble
Target AUC: 0.782+ (Top 1-3% of GCI cohort)
=============================================================================
"""

import numpy as np
import pandas as pd
import os
import warnings
import time
import pickle

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─────────────────────────────────────────────
# 0. CONFIG
# ─────────────────────────────────────────────
DATA_DIR   = r"c:\Users\DILEEP\Downloads\PLACEMENT_PREP\Course_omnicampus\competition_data"
OUTPUT_DIR = r"c:\Users\DILEEP\Downloads\PLACEMENT_PREP\Course_omnicampus\competition_data\output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_FOLDS    = 5
SEED       = 42

print("=" * 70)
print("GCI World 2026 - Home Credit Default Risk - TOP 1% PIPELINE")
print("=" * 70)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
t0 = time.time()
print("\n[1] Loading data...")
train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")
sample_sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
print(f"   train: {train.shape}, test: {test.shape}")

y_train = train['TARGET'].copy()
train_ids = train['SK_ID_CURR'].copy()
test_ids  = test['SK_ID_CURR'].copy()

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n[2] Feature Engineering (Elite 60+ features)...")

def engineer_features(df):
    """
    Comprehensive feature engineering for Home Credit application table.
    Inspired by top GCI solutions + Kaggle competition winners.
    """
    d = df.copy()

    # ── 2A. DAYS_EMPLOYED ANOMALY FIX ──────────────────────────────────
    # 365243 = anomalous code for retired/unemployed — critical fix
    d['DAYS_EMPLOYED_ANOM'] = (d['DAYS_EMPLOYED'] == 365243).astype(np.int8)
    d['DAYS_EMPLOYED'] = d['DAYS_EMPLOYED'].replace(365243, np.nan)

    # ── 2B. AGE & EMPLOYMENT FEATURES ──────────────────────────────────
    d['AGE_YEARS']        = d['DAYS_BIRTH']      / -365.0
    d['EMPLOYED_YEARS']   = d['DAYS_EMPLOYED']   / -365.0
    d['ID_PUBLISH_YEARS'] = d['DAYS_ID_PUBLISH'] / -365.0
    d['REG_YEARS']        = d['DAYS_REGISTRATION']/ -365.0

    # Employment ratio (years employed / age in years)
    d['EMPLOYED_TO_AGE_RATIO'] = d['EMPLOYED_YEARS'] / (d['AGE_YEARS'] + 0.001)

    # Negative: recent phone change relative to days employed
    d['PHONE_TO_EMPLOY_RATIO'] = (d['DAYS_LAST_PHONE_CHANGE']
                                   / (d['DAYS_EMPLOYED'] + 0.001))

    # Age buckets (known non-linear relationship with default)
    d['AGE_BIN'] = pd.cut(d['AGE_YEARS'],
                           bins=[0, 25, 35, 45, 55, 65, 100],
                           labels=[0, 1, 2, 3, 4, 5]).astype(float)

    # ── 2C. FINANCIAL RATIOS ────────────────────────────────────────────
    # These are the most critical domain-specific features
    d['CREDIT_TO_INCOME']     = d['AMT_CREDIT']      / (d['AMT_INCOME_TOTAL']  + 0.001)
    d['ANNUITY_TO_INCOME']    = d['AMT_ANNUITY']     / (d['AMT_INCOME_TOTAL']  + 0.001)
    d['CREDIT_TO_GOODS']      = d['AMT_CREDIT']      / (d['AMT_GOODS_PRICE']   + 0.001)
    d['ANNUITY_TO_CREDIT']    = d['AMT_ANNUITY']     / (d['AMT_CREDIT']        + 0.001)
    d['INCOME_PER_PERSON']    = d['AMT_INCOME_TOTAL'] / (d['CNT_FAM_MEMBERS']  + 0.001)
    d['CREDIT_TERM']          = d['AMT_CREDIT']      / (d['AMT_ANNUITY']       + 0.001)
    d['GOODS_DEFICIT']        = d['AMT_GOODS_PRICE'] - d['AMT_CREDIT']

    # Children ratios
    d['CHILDREN_TO_INCOME']  = d['CNT_CHILDREN']    / (d['AMT_INCOME_TOTAL']  + 0.001)
    d['CHILDREN_RATIO']      = d['CNT_CHILDREN']    / (d['CNT_FAM_MEMBERS']   + 0.001)

    # ── 2D. EXT_SOURCE INTERACTIONS ─────────────────────────────────────
    # These external credit bureau scores are THE most predictive signals
    # Missing rates: EXT_SOURCE_1=69%, EXT_SOURCE_2=0.2%, EXT_SOURCE_3=32%
    
    # Mean across available sources (handles NaN gracefully)
    d['EXT_SRC_MEAN']  = d[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].mean(axis=1)
    d['EXT_SRC_STD']   = d[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].std(axis=1)
    d['EXT_SRC_MIN']   = d[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].min(axis=1)
    d['EXT_SRC_MAX']   = d[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].max(axis=1)
    d['EXT_SRC_RANGE'] = d['EXT_SRC_MAX'] - d['EXT_SRC_MIN']

    # Pairwise products (capture joint non-linear risk)
    d['EXT_12_PROD'] = d['EXT_SOURCE_1'] * d['EXT_SOURCE_2']
    d['EXT_13_PROD'] = d['EXT_SOURCE_1'] * d['EXT_SOURCE_3']
    d['EXT_23_PROD'] = d['EXT_SOURCE_2'] * d['EXT_SOURCE_3']

    # EXT combined with age — high score + young = lower default risk
    d['EXT_SRC_MEAN_X_AGE']      = d['EXT_SRC_MEAN'] * d['AGE_YEARS']
    d['EXT_SRC_MEAN_X_EMPLOYED'] = d['EXT_SRC_MEAN'] * d['EMPLOYED_YEARS'].clip(lower=0)

    # EXT2 squared (confirmed non-linear relationship in past analysis)
    d['EXT_SOURCE_2_SQ'] = d['EXT_SOURCE_2'] ** 2
    d['EXT_SOURCE_3_SQ'] = d['EXT_SOURCE_3'] ** 2

    # EXT combined with credit ratios
    d['EXT_SRC_X_CREDIT_RATIO']   = d['EXT_SRC_MEAN'] * d['CREDIT_TO_INCOME']
    d['EXT_SRC_X_ANNUITY_RATIO']  = d['EXT_SRC_MEAN'] * d['ANNUITY_TO_INCOME']

    # ── 2E. DOCUMENT & CONTACT FLAGS ────────────────────────────────────
    # Sum of contact flags provided
    contact_cols = ['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE']
    existing_contact = [c for c in contact_cols if c in d.columns]
    d['CONTACT_COUNT'] = d[existing_contact].sum(axis=1)

    # ── 2F. CREDIT BUREAU REQUEST FEATURES ──────────────────────────────
    bureau_req_cols = ['AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_MON',
                       'AMT_REQ_CREDIT_BUREAU_QRT',  'AMT_REQ_CREDIT_BUREAU_YEAR']
    existing_bureau = [c for c in bureau_req_cols if c in d.columns]
    if existing_bureau:
        d['TOTAL_CREDIT_INQUIRIES'] = d[existing_bureau].fillna(0).sum(axis=1)
        # High inquiries in last month relative to year = recently desperate
        if 'AMT_REQ_CREDIT_BUREAU_MON' in d.columns and 'AMT_REQ_CREDIT_BUREAU_YEAR' in d.columns:
            d['RECENT_INQUIRY_RATIO'] = (
                d['AMT_REQ_CREDIT_BUREAU_MON'].fillna(0) /
                (d['AMT_REQ_CREDIT_BUREAU_YEAR'].fillna(0) + 1)
            )

    # ── 2G. MISSING VALUE PATTERN ───────────────────────────────────────
    # The pattern of missing values itself carries information
    d['MISSING_COUNT'] = d.isnull().sum(axis=1)

    # Flag: whether EXT_SOURCE_1 is available (proxy for credit history depth)
    d['HAS_EXT_SOURCE_1'] = (~d['EXT_SOURCE_1'].isnull()).astype(np.int8)
    d['HAS_EXT_SOURCE_3'] = (~d['EXT_SOURCE_3'].isnull()).astype(np.int8)
    d['HAS_OWN_CAR_AGE']  = (~d['OWN_CAR_AGE'].isnull()).astype(np.int8)

    # ── 2H. OWN_CAR_AGE FEATURES ────────────────────────────────────────
    # OWN_CAR_AGE >= 60 is likely anomalous (old data entry)
    d['OWN_CAR_AGE_ANOM'] = (d['OWN_CAR_AGE'] >= 60).astype(np.int8)
    d['OWN_CAR_AGE_CLN']  = d['OWN_CAR_AGE'].clip(upper=59)

    # ── 2I. REGION FEATURES ─────────────────────────────────────────────
    if 'REGION_RATING_CLIENT' in d.columns and 'REGION_POPULATION_RELATIVE' in d.columns:
        d['REGION_SCORE'] = (d['REGION_RATING_CLIENT']
                             * d['REGION_POPULATION_RELATIVE'])

    print(f"   -> Feature count: {d.shape[1]} columns (from {df.shape[1]} raw)")
    return d


# Apply feature engineering to train and test
all_data = pd.concat([train.drop('TARGET', axis=1), test], ignore_index=True)
all_eng  = engineer_features(all_data)

train_eng = all_eng.iloc[:len(train)].copy()
test_eng  = all_eng.iloc[len(train):].copy()

# ─────────────────────────────────────────────
# 3. ENCODE CATEGORICAL FEATURES
# ─────────────────────────────────────────────
print("\n[3] Encoding categorical features...")

cat_cols = [c for c in train_eng.columns if train_eng[c].dtype == 'object']
print(f"   Categorical columns: {cat_cols}")

for col in cat_cols:
    # Label-encode with frequency-based mapping
    freq = train_eng[col].value_counts().to_dict()
    train_eng[col] = train_eng[col].map(freq).fillna(0)
    test_eng[col]  = test_eng[col].map(freq).fillna(0)
    print(f"   Encoded {col}: {len(freq)} unique values")

# ─────────────────────────────────────────────
# 4. PREPARE FINAL FEATURE SET
# ─────────────────────────────────────────────
print("\n[4] Preparing feature sets...")

# Drop ID column from features
drop_cols = ['SK_ID_CURR']
feature_cols = [c for c in train_eng.columns if c not in drop_cols]

X      = train_eng[feature_cols].values
X_test = test_eng[feature_cols].values
y      = y_train.values

print(f"   X shape      : {X.shape}")
print(f"   X_test shape : {X_test.shape}")
print(f"   y shape      : {y.shape}, positive rate: {y.mean()*100:.2f}%")

# ─────────────────────────────────────────────
# 5. LIGHTGBM 5-FOLD STRATIFIED CV
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("[5] LightGBM 5-Fold Stratified CV")
print("=" * 70)

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Elite LightGBM params tuned for this data size/class imbalance
LGBM_PARAMS = {
    'objective':         'binary',
    'metric':            'auc',
    'boosting_type':     'gbdt',
    'learning_rate':     0.02,
    'num_leaves':        63,
    'max_depth':         -1,
    'min_child_samples': 30,
    'feature_fraction':  0.80,
    'bagging_fraction':  0.80,
    'bagging_freq':      5,
    'reg_alpha':         0.1,
    'reg_lambda':        0.1,
    'scale_pos_weight':  (y == 0).sum() / (y == 1).sum(),  # handle imbalance
    'n_estimators':      5000,
    'verbose':           -1,
    'random_state':      SEED,
    'n_jobs':            -1,
}

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_lgbm   = np.zeros(len(X))
pred_lgbm  = np.zeros(len(X_test))
lgbm_scores = []
feature_names = feature_cols

t_lgbm = time.time()
for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y)):
    X_trn, X_val = X[trn_idx], X[val_idx]
    y_trn, y_val = y[trn_idx], y[val_idx]

    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(
        X_trn, y_trn,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(500),
        ]
    )

    oof_val = model.predict_proba(X_val)[:, 1]
    oof_lgbm[val_idx] = oof_val

    test_pred = model.predict_proba(X_test)[:, 1]
    pred_lgbm += test_pred / N_FOLDS

    fold_auc = roc_auc_score(y_val, oof_val)
    lgbm_scores.append(fold_auc)
    print(f"   Fold {fold+1}/{N_FOLDS} | AUC: {fold_auc:.5f} | Best iter: {model.best_iteration_}")

lgbm_oof_auc = roc_auc_score(y, oof_lgbm)
lgbm_elapsed = time.time() - t_lgbm
print(f"\n   LightGBM OOF AUC: {lgbm_oof_auc:.5f}")
print(f"   Fold AUCs: {[f'{s:.5f}' for s in lgbm_scores]}")
print(f"   CV Mean ± Std: {np.mean(lgbm_scores):.5f} ± {np.std(lgbm_scores):.5f}")
print(f"   Time elapsed: {lgbm_elapsed/60:.1f} min")

# Save feature importance from last model
fi_df = pd.DataFrame({
    'feature':    feature_names,
    'importance': model.feature_importances_,
}).sort_values('importance', ascending=False)
fi_df.to_csv(f"{OUTPUT_DIR}/lgbm_feature_importance.csv", index=False)
print("\nTop 20 Features by LightGBM importance:")
print(fi_df.head(20).to_string(index=False))

# ─────────────────────────────────────────────
# 6. CATBOOST 5-FOLD STRATIFIED CV
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("[6] CatBoost 5-Fold Stratified CV")
print("=" * 70)

from catboost import CatBoostClassifier

CATBOOST_PARAMS = {
    'iterations':       3000,
    'learning_rate':    0.03,
    'depth':            7,
    'l2_leaf_reg':      3.0,
    'bootstrap_type':   'Bayesian',
    'bagging_temperature': 1.0,
    'colsample_bylevel': 0.8,
    'eval_metric':      'AUC',
    'scale_pos_weight': (y == 0).sum() / (y == 1).sum(),
    'random_seed':      SEED,
    'verbose':          False,
    'early_stopping_rounds': 100,
    'task_type':        'CPU',
}

oof_cat   = np.zeros(len(X))
pred_cat  = np.zeros(len(X_test))
cat_scores = []

t_cat = time.time()
for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y)):
    X_trn, X_val = X[trn_idx], X[val_idx]
    y_trn, y_val = y[trn_idx], y[val_idx]

    model_cb = CatBoostClassifier(**CATBOOST_PARAMS)
    model_cb.fit(
        X_trn, y_trn,
        eval_set=(X_val, y_val),
        use_best_model=True,
    )

    oof_val = model_cb.predict_proba(X_val)[:, 1]
    oof_cat[val_idx] = oof_val

    test_pred = model_cb.predict_proba(X_test)[:, 1]
    pred_cat += test_pred / N_FOLDS

    fold_auc = roc_auc_score(y_val, oof_val)
    cat_scores.append(fold_auc)
    print(f"   Fold {fold+1}/{N_FOLDS} | AUC: {fold_auc:.5f} | Best iter: {model_cb.best_iteration_}")

cat_oof_auc = roc_auc_score(y, oof_cat)
cat_elapsed = time.time() - t_cat
print(f"\n   CatBoost OOF AUC: {cat_oof_auc:.5f}")
print(f"   Fold AUCs: {[f'{s:.5f}' for s in cat_scores]}")
print(f"   CV Mean ± Std: {np.mean(cat_scores):.5f} ± {np.std(cat_scores):.5f}")
print(f"   Time elapsed: {cat_elapsed/60:.1f} min")

# ─────────────────────────────────────────────
# 7. ENSEMBLE: RANK AVERAGING
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("[7] Ensemble: Rank-Averaged Blending")
print("=" * 70)

from scipy.stats import rankdata

# Rank-average is better than simple average for blending different model families
def rank_avg(preds_list, weights=None):
    if weights is None:
        weights = [1.0] * len(preds_list)
    total_w = sum(weights)
    result = np.zeros(len(preds_list[0]))
    for pred, w in zip(preds_list, weights):
        result += (rankdata(pred) / len(pred)) * (w / total_w)
    return result

# OOF ensemble for selecting best weights
oof_blend = rank_avg([oof_lgbm, oof_cat], weights=[0.55, 0.45])
blend_oof_auc = roc_auc_score(y, oof_blend)
print(f"   LGBM OOF AUC:   {lgbm_oof_auc:.5f}")
print(f"   CatBoost OOF:   {cat_oof_auc:.5f}")
print(f"   Blend OOF AUC:  {blend_oof_auc:.5f}  [LGBM 55% + CatBoost 45%]")

# Also try pure LGBM in case blend doesn't help
print(f"\n   Pure LGBM: {lgbm_oof_auc:.5f}")
print(f"   Pure Cat:  {cat_oof_auc:.5f}")

# Use the best
if blend_oof_auc >= max(lgbm_oof_auc, cat_oof_auc):
    final_test_pred = rank_avg([pred_lgbm, pred_cat], weights=[0.55, 0.45])
    strategy = "Blend(LGBM+CatBoost)"
    final_oof_auc  = blend_oof_auc
elif lgbm_oof_auc >= cat_oof_auc:
    final_test_pred = pred_lgbm
    strategy = "LightGBM only"
    final_oof_auc  = lgbm_oof_auc
else:
    final_test_pred = pred_cat
    strategy = "CatBoost only"
    final_oof_auc  = cat_oof_auc

print(f"\n   Best strategy: {strategy} | Final OOF AUC: {final_oof_auc:.5f}")

# ─────────────────────────────────────────────
# 8. CREATE SUBMISSION
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("[8] Creating Submission File")
print("=" * 70)

submission = sample_sub.copy()
submission['TARGET'] = final_test_pred

sub_path = f"{OUTPUT_DIR}/submission.csv"
submission.to_csv(sub_path, index=False)
print(f"   Saved: {sub_path}")
print(f"   Shape: {submission.shape}")
print(f"   Prediction range: [{final_test_pred.min():.4f}, {final_test_pred.max():.4f}]")
print(f"   Mean prediction: {final_test_pred.mean():.4f}")
print(f"\n   First 5 rows:")
print(submission.head().to_string(index=False))

# ─────────────────────────────────────────────
# 9. FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
total_time = time.time() - t0
print(f"  LightGBM OOF AUC  : {lgbm_oof_auc:.5f}")
print(f"  CatBoost  OOF AUC : {cat_oof_auc:.5f}")
print(f"  Ensemble  OOF AUC : {blend_oof_auc:.5f}")
print(f"  Strategy used     : {strategy}")
print(f"  Final OOF AUC     : {final_oof_auc:.5f}")
print(f"  Total runtime     : {total_time/60:.1f} min")
print(f"  Submission saved  : {sub_path}")
print("=" * 70)
print("READY FOR SUBMISSION TO OMNICAMPUS!")
print("=" * 70)
