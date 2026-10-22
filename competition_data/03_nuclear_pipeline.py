"""
=============================================================================
GCI World 2026 — Home Credit Default Risk
NUCLEAR PIPELINE v3 — Target: Beat 0.79427
=============================================================================
Strategy:
  1. Hyper-aggressive feature engineering (200+ features)
  2. Target encoding with CV leak prevention
  3. Optimized LightGBM (lower LR, more trees, better params)
  4. Pseudo-labeling (high-confidence test rows added to training)
  5. XGBoost 5-Fold added to ensemble
  6. 3-way rank-average blend (LGBM + CatBoost + XGB)
  7. Re-train on FULL data (train + pseudo-labeled) for final predictions
=============================================================================
"""
import pandas as pd
import numpy as np
import time
import warnings
import os
warnings.filterwarnings('ignore')

SEED       = 42
N_FOLDS    = 5
DATA_DIR   = r"c:\Users\DILEEP\Downloads\PLACEMENT_PREP\Course_omnicampus\competition_data"
OUTPUT_DIR = r"c:\Users\DILEEP\Downloads\PLACEMENT_PREP\Course_omnicampus\competition_data\output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(SEED)
t0 = time.time()

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 70)
print("[1] Loading data...")
print("=" * 70)
train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")
sample_sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

print(f"   train: {train.shape}, test: {test.shape}")

# ─────────────────────────────────────────────
# 2. HYPER-AGGRESSIVE FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("[2] Hyper-Aggressive Feature Engineering (200+ features)...")
print("=" * 70)

def engineer_features(df):
    d = df.copy()

    # ── DAYS_EMPLOYED anomaly ──────────────────────────────────────
    d['DAYS_EMPLOYED_ANOM'] = (d['DAYS_EMPLOYED'] == 365243).astype(int)
    d['DAYS_EMPLOYED_CLN']  = d['DAYS_EMPLOYED'].replace(365243, np.nan)

    # ── AGE / EMPLOYMENT ──────────────────────────────────────────
    d['AGE_YEARS']              = -d['DAYS_BIRTH'] / 365.25
    d['EMPLOYED_YEARS']         = -d['DAYS_EMPLOYED_CLN'] / 365.25
    d['EMPLOYED_TO_AGE_RATIO']  = d['EMPLOYED_YEARS'] / (d['AGE_YEARS'] + 1e-8)
    d['AGE_SQ']                 = d['AGE_YEARS'] ** 2
    d['EMPLOYED_SQ']            = d['EMPLOYED_YEARS'].fillna(0) ** 2

    # ── ID / PHONE CHANGE ─────────────────────────────────────────
    d['ID_PUBLISH_YEARS']    = -d['DAYS_ID_PUBLISH'] / 365.25
    d['REG_YEARS']           = -d['DAYS_REGISTRATION'] / 365.25
    d['PHONE_CHANGE_YEARS']  = -d['DAYS_LAST_PHONE_CHANGE'] / 365.25
    d['PHONE_TO_BIRTH_RATIO']= d['DAYS_LAST_PHONE_CHANGE'] / (d['DAYS_BIRTH'] - 1e-8)
    d['PHONE_TO_EMPLOY_RATIO']= d['DAYS_LAST_PHONE_CHANGE'] / (d['DAYS_EMPLOYED_CLN'] - 1e-8)
    d['ID_TO_AGE_RATIO']     = d['DAYS_ID_PUBLISH'] / (d['DAYS_BIRTH'] - 1e-8)
    d['REG_TO_AGE_RATIO']    = d['DAYS_REGISTRATION'] / (d['DAYS_BIRTH'] - 1e-8)

    # ── FINANCIAL RATIOS ─────────────────────────────────────────
    d['CREDIT_TO_INCOME']    = d['AMT_CREDIT'] / (d['AMT_INCOME_TOTAL'] + 1e-8)
    d['ANNUITY_TO_INCOME']   = d['AMT_ANNUITY'] / (d['AMT_INCOME_TOTAL'] + 1e-8)
    d['ANNUITY_TO_CREDIT']   = d['AMT_ANNUITY'] / (d['AMT_CREDIT'] + 1e-8)
    d['CREDIT_TO_GOODS']     = d['AMT_CREDIT'] / (d['AMT_GOODS_PRICE'] + 1e-8)
    d['INCOME_PER_PERSON']   = d['AMT_INCOME_TOTAL'] / (d['CNT_FAM_MEMBERS'] + 1)
    d['CHILDREN_TO_INCOME']  = d['CNT_CHILDREN'] / (d['AMT_INCOME_TOTAL'] + 1e-8)
    d['CHILDREN_RATIO']      = d['CNT_CHILDREN'] / (d['CNT_FAM_MEMBERS'] + 1)
    d['CREDIT_TERM']         = d['AMT_ANNUITY'] / (d['AMT_CREDIT'] + 1e-8)
    d['GOODS_DEFICIT']       = d['AMT_CREDIT'] - d['AMT_GOODS_PRICE']
    d['GOODS_DEFICIT_RATIO'] = d['GOODS_DEFICIT'] / (d['AMT_GOODS_PRICE'] + 1e-8)
    d['INCOME_CREDIT_PCT']   = d['AMT_INCOME_TOTAL'] / (d['AMT_CREDIT'] + 1e-8)
    d['ANNUITY_TO_GOODS']    = d['AMT_ANNUITY'] / (d['AMT_GOODS_PRICE'] + 1e-8)
    d['INCOME_SQ_RATIO']     = d['AMT_INCOME_TOTAL'] ** 2 / (d['AMT_CREDIT'] + 1e-8)

    # ── EXT_SOURCE FEATURES (THE MOST IMPORTANT) ─────────────────
    d['EXT_SRC_MEAN']  = d[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].mean(axis=1)
    d['EXT_SRC_MAX']   = d[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].max(axis=1)
    d['EXT_SRC_MIN']   = d[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].min(axis=1)
    d['EXT_SRC_STD']   = d[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].std(axis=1)
    d['EXT_SRC_RANGE'] = d['EXT_SRC_MAX'] - d['EXT_SRC_MIN']
    d['EXT_SRC_SUM']   = d[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].sum(axis=1)
    d['EXT_SRC_PROD']  = (d['EXT_SOURCE_1'].fillna(0) *
                          d['EXT_SOURCE_2'].fillna(0) *
                          d['EXT_SOURCE_3'].fillna(0))

    # Pairwise products
    d['EXT_12_PROD']   = d['EXT_SOURCE_1'].fillna(0) * d['EXT_SOURCE_2'].fillna(0)
    d['EXT_13_PROD']   = d['EXT_SOURCE_1'].fillna(0) * d['EXT_SOURCE_3'].fillna(0)
    d['EXT_23_PROD']   = d['EXT_SOURCE_2'].fillna(0) * d['EXT_SOURCE_3'].fillna(0)
    
    # Pairwise differences
    d['EXT_12_DIFF']   = d['EXT_SOURCE_1'].fillna(0) - d['EXT_SOURCE_2'].fillna(0)
    d['EXT_13_DIFF']   = d['EXT_SOURCE_1'].fillna(0) - d['EXT_SOURCE_3'].fillna(0)
    d['EXT_23_DIFF']   = d['EXT_SOURCE_2'].fillna(0) - d['EXT_SOURCE_3'].fillna(0)

    # Polynomial
    d['EXT_SOURCE_1_SQ']   = d['EXT_SOURCE_1'] ** 2
    d['EXT_SOURCE_2_SQ']   = d['EXT_SOURCE_2'] ** 2
    d['EXT_SOURCE_3_SQ']   = d['EXT_SOURCE_3'] ** 2
    d['EXT_SOURCE_1_CB']   = d['EXT_SOURCE_1'] ** 3
    d['EXT_SOURCE_2_CB']   = d['EXT_SOURCE_2'] ** 3
    d['EXT_SOURCE_3_CB']   = d['EXT_SOURCE_3'] ** 3
    d['EXT_SOURCE_1_SQRT'] = np.sqrt(d['EXT_SOURCE_1'].clip(0))
    d['EXT_SOURCE_2_SQRT'] = np.sqrt(d['EXT_SOURCE_2'].clip(0))
    d['EXT_SOURCE_3_SQRT'] = np.sqrt(d['EXT_SOURCE_3'].clip(0))
    d['EXT_SRC_MEAN_SQ']   = d['EXT_SRC_MEAN'] ** 2

    # EXT × financial interactions
    d['EXT_SRC_MEAN_X_AGE']      = d['EXT_SRC_MEAN'] * d['AGE_YEARS']
    d['EXT_SRC_MEAN_X_EMPLOYED'] = d['EXT_SRC_MEAN'] * d['EMPLOYED_YEARS'].fillna(0)
    d['EXT_SRC_X_CREDIT_RATIO']  = d['EXT_SRC_MEAN'] / (d['CREDIT_TO_INCOME'] + 1e-8)
    d['EXT_SRC_X_ANNUITY_RATIO'] = d['EXT_SRC_MEAN'] / (d['ANNUITY_TO_INCOME'] + 1e-8)
    d['EXT2_X_INCOME']           = d['EXT_SOURCE_2'] * d['AMT_INCOME_TOTAL']
    d['EXT3_X_CREDIT']           = d['EXT_SOURCE_3'] * d['AMT_CREDIT']
    d['EXT2_X_ANNUITY']          = d['EXT_SOURCE_2'] * d['AMT_ANNUITY'].fillna(0)
    d['EXT3_X_GOODS']            = d['EXT_SOURCE_3'] * d['AMT_GOODS_PRICE'].fillna(0)
    d['EXT_MEAN_X_CREDIT_INCOME']= d['EXT_SRC_MEAN'] * d['CREDIT_TO_INCOME']

    # ── CAR AGE ───────────────────────────────────────────────────
    d['OWN_CAR_AGE_ANOM'] = (d['OWN_CAR_AGE'] > 60).astype(int)
    d['OWN_CAR_AGE_CLN']  = d['OWN_CAR_AGE'].clip(0, 60)
    d['HAS_OWN_CAR_AGE']  = d['OWN_CAR_AGE'].notna().astype(int)

    # ── CREDIT BUREAU INQUIRIES ───────────────────────────────────
    d['TOTAL_CREDIT_INQUIRIES']   = (d['AMT_REQ_CREDIT_BUREAU_HOUR'].fillna(0) +
                                     d['AMT_REQ_CREDIT_BUREAU_MON'].fillna(0) +
                                     d['AMT_REQ_CREDIT_BUREAU_QRT'].fillna(0) +
                                     d['AMT_REQ_CREDIT_BUREAU_YEAR'].fillna(0))
    d['RECENT_INQUIRY_RATIO'] = (d['AMT_REQ_CREDIT_BUREAU_HOUR'].fillna(0) + 1) / \
                                 (d['AMT_REQ_CREDIT_BUREAU_YEAR'].fillna(0) + 1)
    d['INQUIRY_ACCELERATION'] = d['AMT_REQ_CREDIT_BUREAU_QRT'].fillna(0) / \
                                 (d['AMT_REQ_CREDIT_BUREAU_YEAR'].fillna(0) + 1)

    # ── CONTACT FLAGS ─────────────────────────────────────────────
    d['CONTACT_COUNT']   = d['FLAG_MOBIL'] + d['FLAG_EMP_PHONE'] + d['FLAG_WORK_PHONE']

    # ── MISSING VALUE COUNT ───────────────────────────────────────
    num_cols = d.select_dtypes(include='number').columns.tolist()
    d['MISSING_COUNT'] = d[num_cols].isnull().sum(axis=1)

    # ── HAS FLAGS ─────────────────────────────────────────────────
    d['HAS_EXT_SOURCE_1'] = d['EXT_SOURCE_1'].notna().astype(int)
    d['HAS_EXT_SOURCE_3'] = d['EXT_SOURCE_3'].notna().astype(int)

    # ── REGION INTERACTION ────────────────────────────────────────
    d['REGION_SCORE']    = d['REGION_RATING_CLIENT'] * d['REGION_POPULATION_RELATIVE']
    d['REGION_X_EXT2']   = d['REGION_RATING_CLIENT'] * d['EXT_SOURCE_2'].fillna(0)
    d['REGION_X_AGE']    = d['REGION_RATING_CLIENT'] * d['AGE_YEARS']

    # ── AGE BINS ─────────────────────────────────────────────────
    d['AGE_BIN'] = pd.cut(d['AGE_YEARS'], bins=[0,25,35,45,55,65,100], labels=False)

    # ── INCOME BINS ───────────────────────────────────────────────
    d['INCOME_BIN'] = pd.qcut(d['AMT_INCOME_TOTAL'], q=10, labels=False, duplicates='drop')
    d['CREDIT_BIN']  = pd.qcut(d['AMT_CREDIT'], q=10, labels=False, duplicates='drop')

    # ── EMPLOYMENT RATIO × AGE BIN ───────────────────────────────
    d['EMPLOY_AGE_BIN_INTERACTION'] = d['EMPLOYED_TO_AGE_RATIO'] * d['AGE_BIN'].fillna(0)

    # ── ABSOLUTE DAYS ─────────────────────────────────────────────
    d['ABS_DAYS_BIRTH']            = d['DAYS_BIRTH'].abs()
    d['ABS_DAYS_EMPLOYED_CLN']     = d['DAYS_EMPLOYED_CLN'].abs()
    d['ABS_DAYS_REGISTRATION']     = d['DAYS_REGISTRATION'].abs()
    d['ABS_DAYS_ID_PUBLISH']       = d['DAYS_ID_PUBLISH'].abs()
    d['ABS_DAYS_LAST_PHONE_CHANGE']= d['DAYS_LAST_PHONE_CHANGE'].abs()

    print(f"   -> Feature count: {d.shape[1]} columns (from {df.shape[1]} raw)")
    return d

all_data = pd.concat([train.drop('TARGET', axis=1), test], ignore_index=True)
all_eng  = engineer_features(all_data)
train_eng = all_eng.iloc[:len(train)].copy()
test_eng  = all_eng.iloc[len(train):].copy()

# ─────────────────────────────────────────────
# 3. ENCODING
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("[3] Encoding categorical features...")
print("=" * 70)

cat_cols = [c for c in train_eng.select_dtypes('object').columns]
for c in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train_eng[c], test_eng[c]], axis=0).astype(str)
    le.fit(combined)
    train_eng[c] = le.transform(train_eng[c].astype(str))
    test_eng[c]  = le.transform(test_eng[c].astype(str))
    print(f"   Encoded {c}: {train_eng[c].nunique()} unique values")

# ─────────────────────────────────────────────
# 4. TARGET ENCODING (with CV leak prevention)
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("[4] Target Encoding with CV leak prevention...")
print("=" * 70)

y     = train['TARGET'].values
y_ser = train['TARGET']
skf   = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# Target-encode top categorical features
te_cols = ['ORGANIZATION_TYPE', 'OCCUPATION_TYPE', 'NAME_INCOME_TYPE',
           'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
           'NAME_CONTRACT_TYPE', 'CODE_GENDER']

global_mean = y_ser.mean()

for col in te_cols:
    te_train = np.full(len(train_eng), global_mean)
    te_test  = np.zeros(len(test_eng))
    fold_predictions = []
    for tr_idx, va_idx in skf.split(train_eng, y):
        mapping = y_ser.iloc[tr_idx].groupby(train_eng[col].iloc[tr_idx]).mean()
        te_train[va_idx] = train_eng[col].iloc[va_idx].map(mapping).fillna(global_mean)
        fold_predictions.append(train_eng[col].map(mapping).fillna(global_mean).values)
    te_test = np.mean(fold_predictions, axis=0)[len(train_eng):]  # doesn't make sense
    # For test: use full train mapping
    full_mapping = y_ser.groupby(train_eng[col]).mean()
    te_test      = test_eng[col].map(full_mapping).fillna(global_mean).values
    train_eng[f'{col}_TE'] = te_train
    test_eng[f'{col}_TE']  = te_test
    print(f"   Target-encoded: {col}")

# ─────────────────────────────────────────────
# 5. PREPARE FEATURE MATRICES
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("[5] Preparing feature matrices...")
print("=" * 70)

drop_cols    = ['SK_ID_CURR', 'TARGET'] if 'TARGET' in train_eng.columns else ['SK_ID_CURR']
feature_cols = [c for c in train_eng.columns if c not in drop_cols]

X      = train_eng[feature_cols].values.astype(np.float32)
X_test = test_eng[feature_cols].values.astype(np.float32)
print(f"   X shape:      {X.shape}")
print(f"   X_test shape: {X_test.shape}")
print(f"   Positive rate: {y.mean():.2%}")

# ─────────────────────────────────────────────
# 6. LIGHTGBM — AGGRESSIVE TUNING
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("[6] LightGBM 5-Fold CV (Aggressive tuning)")
print("=" * 70)

import lightgbm as lgb

LGBM_PARAMS = {
    'objective':        'binary',
    'metric':           'auc',
    'boosting_type':    'gbdt',
    'n_estimators':     10000,
    'learning_rate':    0.01,          # Lower LR = more trees = better
    'num_leaves':       63,            # Up from 31
    'max_depth':        -1,
    'min_child_samples':20,
    'min_child_weight': 0.001,
    'colsample_bytree': 0.5,
    'subsample':        0.8,
    'subsample_freq':   1,
    'reg_alpha':        0.1,
    'reg_lambda':       0.1,
    'scale_pos_weight': (y == 0).sum() / (y == 1).sum(),
    'random_state':     SEED,
    'n_jobs':           -1,
    'verbose':          -1,
}

oof_lgbm   = np.zeros(len(X))
pred_lgbm  = np.zeros(len(X_test))
lgbm_scores = []
t_lgbm = time.time()

for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y)):
    X_trn, X_val = X[trn_idx], X[val_idx]
    y_trn, y_val = y[trn_idx], y[val_idx]

    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(
        X_trn, y_trn,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(150, verbose=False),
                   lgb.log_evaluation(500)],
    )

    oof_lgbm[val_idx] = model.predict_proba(X_val)[:, 1]
    pred_lgbm         += model.predict_proba(X_test)[:, 1] / N_FOLDS

    fold_auc = roc_auc_score(y_val, oof_lgbm[val_idx])
    lgbm_scores.append(fold_auc)
    print(f"   Fold {fold+1}/{N_FOLDS} | AUC: {fold_auc:.5f} | Best iter: {model.best_iteration_}")

lgbm_oof_auc = roc_auc_score(y, oof_lgbm)
print(f"\n   LightGBM OOF AUC: {lgbm_oof_auc:.5f}")
print(f"   CV Mean ± Std:    {np.mean(lgbm_scores):.5f} ± {np.std(lgbm_scores):.5f}")
print(f"   Time: {(time.time()-t_lgbm)/60:.1f} min")

# Save feature importance
fi_df = pd.DataFrame({'feature': feature_cols, 'importance': model.feature_importances_})
fi_df = fi_df.sort_values('importance', ascending=False)
fi_df.to_csv(f"{OUTPUT_DIR}/v3_lgbm_feature_importance.csv", index=False)
print("\nTop 15 features:")
print(fi_df.head(15).to_string(index=False))

# ─────────────────────────────────────────────
# 7. CATBOOST 5-FOLD
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("[7] CatBoost 5-Fold CV")
print("=" * 70)

from catboost import CatBoostClassifier

CATBOOST_PARAMS = {
    'iterations':           5000,
    'learning_rate':        0.02,
    'depth':                7,
    'l2_leaf_reg':          3.0,
    'bootstrap_type':       'Bayesian',
    'bagging_temperature':  0.8,
    'colsample_bylevel':    0.7,
    'eval_metric':          'AUC',
    'scale_pos_weight':     (y == 0).sum() / (y == 1).sum(),
    'random_seed':          SEED,
    'verbose':              False,
    'early_stopping_rounds':150,
    'task_type':            'CPU',
    'od_type':              'Iter',
}

oof_cat   = np.zeros(len(X))
pred_cat  = np.zeros(len(X_test))
cat_scores = []
t_cat = time.time()

for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y)):
    X_trn, X_val = X[trn_idx], X[val_idx]
    y_trn, y_val = y[trn_idx], y[val_idx]

    model_cb = CatBoostClassifier(**CATBOOST_PARAMS)
    model_cb.fit(X_trn, y_trn, eval_set=(X_val, y_val), use_best_model=True)

    oof_cat[val_idx]  = model_cb.predict_proba(X_val)[:, 1]
    pred_cat         += model_cb.predict_proba(X_test)[:, 1] / N_FOLDS

    fold_auc = roc_auc_score(y_val, oof_cat[val_idx])
    cat_scores.append(fold_auc)
    print(f"   Fold {fold+1}/{N_FOLDS} | AUC: {fold_auc:.5f} | Best iter: {model_cb.best_iteration_}")

cat_oof_auc = roc_auc_score(y, oof_cat)
print(f"\n   CatBoost OOF AUC: {cat_oof_auc:.5f}")
print(f"   CV Mean ± Std:    {np.mean(cat_scores):.5f} ± {np.std(cat_scores):.5f}")
print(f"   Time: {(time.time()-t_cat)/60:.1f} min")

# ─────────────────────────────────────────────
# 8. XGBOOST 5-FOLD
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("[8] XGBoost 5-Fold CV")
print("=" * 70)

try:
    import xgboost as xgb
    XGB_PARAMS = {
        'n_estimators':     10000,
        'learning_rate':    0.01,
        'max_depth':        6,
        'min_child_weight': 5,
        'subsample':        0.8,
        'colsample_bytree': 0.5,
        'colsample_bylevel':0.5,
        'reg_alpha':        0.1,
        'reg_lambda':       1.0,
        'scale_pos_weight': (y == 0).sum() / (y == 1).sum(),
        'eval_metric':      'auc',
        'use_label_encoder':False,
        'random_state':     SEED,
        'n_jobs':           -1,
        'verbosity':        0,
        'tree_method':      'hist',
    }

    oof_xgb   = np.zeros(len(X))
    pred_xgb  = np.zeros(len(X_test))
    xgb_scores = []
    t_xgb = time.time()

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y)):
        X_trn, X_val = X[trn_idx], X[val_idx]
        y_trn, y_val = y[trn_idx], y[val_idx]

        model_xgb = xgb.XGBClassifier(**XGB_PARAMS)
        model_xgb.fit(X_trn, y_trn,
                      eval_set=[(X_val, y_val)],
                      early_stopping_rounds=150,
                      verbose=False)

        oof_xgb[val_idx]  = model_xgb.predict_proba(X_val)[:, 1]
        pred_xgb         += model_xgb.predict_proba(X_test)[:, 1] / N_FOLDS

        fold_auc = roc_auc_score(y_val, oof_xgb[val_idx])
        xgb_scores.append(fold_auc)
        print(f"   Fold {fold+1}/{N_FOLDS} | AUC: {fold_auc:.5f} | Best iter: {model_xgb.best_iteration}")

    xgb_oof_auc = roc_auc_score(y, oof_xgb)
    print(f"\n   XGBoost OOF AUC: {xgb_oof_auc:.5f}")
    print(f"   CV Mean ± Std:   {np.mean(xgb_scores):.5f} ± {np.std(xgb_scores):.5f}")
    print(f"   Time: {(time.time()-t_xgb)/60:.1f} min")
    USE_XGB = True
except Exception as e:
    print(f"XGBoost failed: {e}. Skipping.")
    USE_XGB = False
    oof_xgb  = np.zeros(len(X))
    pred_xgb = np.zeros(len(X_test))
    xgb_oof_auc = 0.0

# ─────────────────────────────────────────────
# 9. PSEUDO-LABELING
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("[9] Pseudo-Labeling (high-confidence test samples)")
print("=" * 70)

# Use blend predictions to identify high-confidence test samples
from scipy.stats import rankdata

def rank_avg(preds_list, weights=None):
    if weights is None:
        weights = [1.0] * len(preds_list)
    total_w = sum(weights)
    result = np.zeros(len(preds_list[0]))
    for pred, w in zip(preds_list, weights):
        result += (rankdata(pred) / len(pred)) * (w / total_w)
    return result

if USE_XGB:
    interim_test_blend = rank_avg([pred_lgbm, pred_cat, pred_xgb], [0.5, 0.25, 0.25])
    interim_oof_blend  = rank_avg([oof_lgbm, oof_cat, oof_xgb], [0.5, 0.25, 0.25])
else:
    interim_test_blend = rank_avg([pred_lgbm, pred_cat], [0.55, 0.45])
    interim_oof_blend  = rank_avg([oof_lgbm, oof_cat], [0.55, 0.45])

# Select very high confidence: top 5% as 1 (default) and bottom 5% as 0 (no default)
CONF_THRESHOLD_HIGH = 0.85
CONF_THRESHOLD_LOW  = 0.15

high_conf_default_idx = np.where(interim_test_blend > CONF_THRESHOLD_HIGH)[0]
high_conf_safe_idx    = np.where(interim_test_blend < CONF_THRESHOLD_LOW)[0]
pseudo_idx            = np.concatenate([high_conf_default_idx, high_conf_safe_idx])
pseudo_labels         = np.concatenate([
    np.ones(len(high_conf_default_idx)),
    np.zeros(len(high_conf_safe_idx))
])

print(f"   High-confidence defaulters (>{CONF_THRESHOLD_HIGH}): {len(high_conf_default_idx)}")
print(f"   High-confidence safe    (<{CONF_THRESHOLD_LOW}): {len(high_conf_safe_idx)}")
print(f"   Total pseudo-labeled:    {len(pseudo_idx)}")

if len(pseudo_idx) > 100:
    X_pseudo = X_test[pseudo_idx]
    y_pseudo = pseudo_labels

    # Augment training data
    X_aug = np.vstack([X, X_pseudo])
    y_aug = np.concatenate([y, y_pseudo])
    print(f"   Augmented train size: {X_aug.shape}")

    # Retrain LGBM on augmented data for final test predictions
    print("   Retraining LightGBM on augmented (train + pseudo-label) data...")
    
    oof_lgbm_pl   = np.zeros(len(X))
    pred_lgbm_pl  = np.zeros(len(X_test))
    lgbm_pl_scores = []
    
    # Use original folds (only evaluate on original train, not pseudo)
    for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y)):
        # Train on fold train + all pseudo
        X_trn_aug = np.vstack([X[trn_idx], X_pseudo])
        y_trn_aug = np.concatenate([y[trn_idx], y_pseudo])
        X_val_orig = X[val_idx]
        y_val_orig = y[val_idx]
        
        model_pl = lgb.LGBMClassifier(**LGBM_PARAMS)
        model_pl.fit(
            X_trn_aug, y_trn_aug,
            eval_set=[(X_val_orig, y_val_orig)],
            callbacks=[lgb.early_stopping(150, verbose=False),
                       lgb.log_evaluation(1000)],
        )
        
        oof_lgbm_pl[val_idx]  = model_pl.predict_proba(X_val_orig)[:, 1]
        pred_lgbm_pl         += model_pl.predict_proba(X_test)[:, 1] / N_FOLDS
        
        fold_auc = roc_auc_score(y_val_orig, oof_lgbm_pl[val_idx])
        lgbm_pl_scores.append(fold_auc)
        print(f"   PL-Fold {fold+1}/{N_FOLDS} | AUC: {fold_auc:.5f}")
    
    lgbm_pl_oof_auc = roc_auc_score(y, oof_lgbm_pl)
    print(f"\n   LGBM+PseudoLabel OOF AUC: {lgbm_pl_oof_auc:.5f}")
    
    # Use pseudo-label version if it's better
    if lgbm_pl_oof_auc > lgbm_oof_auc:
        print("   ✓ Pseudo-label improved LGBM! Using PL predictions.")
        oof_lgbm_final  = oof_lgbm_pl
        pred_lgbm_final = pred_lgbm_pl
    else:
        print("   ✗ Pseudo-label didn't help. Using original predictions.")
        oof_lgbm_final  = oof_lgbm
        pred_lgbm_final = pred_lgbm
else:
    print("   Too few pseudo-labeled samples, skipping.")
    oof_lgbm_final  = oof_lgbm
    pred_lgbm_final = pred_lgbm

# ─────────────────────────────────────────────
# 10. OPTIMAL ENSEMBLE SEARCH
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("[10] Optimal Ensemble Weight Search")
print("=" * 70)

best_auc    = 0.0
best_w_lgbm = 0.5
best_w_cat  = 0.3
best_w_xgb  = 0.2

if USE_XGB:
    for wl in np.arange(0.3, 0.8, 0.05):
        for wc in np.arange(0.1, 0.5, 0.05):
            wx = round(1.0 - wl - wc, 3)
            if wx < 0.05 or wx > 0.5:
                continue
            blend_oof = rank_avg([oof_lgbm_final, oof_cat, oof_xgb], [wl, wc, wx])
            auc = roc_auc_score(y, blend_oof)
            if auc > best_auc:
                best_auc    = auc
                best_w_lgbm = wl
                best_w_cat  = wc
                best_w_xgb  = wx
    print(f"   Best weights: LGBM={best_w_lgbm:.2f} | CatBoost={best_w_cat:.2f} | XGB={best_w_xgb:.2f}")
    print(f"   Best OOF AUC: {best_auc:.5f}")
    final_test_pred = rank_avg([pred_lgbm_final, pred_cat, pred_xgb],
                                [best_w_lgbm, best_w_cat, best_w_xgb])
else:
    for wl in np.arange(0.3, 0.9, 0.05):
        wc = round(1.0 - wl, 3)
        blend_oof = rank_avg([oof_lgbm_final, oof_cat], [wl, wc])
        auc = roc_auc_score(y, blend_oof)
        if auc > best_auc:
            best_auc    = auc
            best_w_lgbm = wl
            best_w_cat  = wc
    print(f"   Best weights: LGBM={best_w_lgbm:.2f} | CatBoost={best_w_cat:.2f}")
    print(f"   Best OOF AUC: {best_auc:.5f}")
    final_test_pred = rank_avg([pred_lgbm_final, pred_cat], [best_w_lgbm, best_w_cat])

# ─────────────────────────────────────────────
# 11. CREATE SUBMISSION
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("[11] Creating Submission File")
print("=" * 70)

submission = sample_sub.copy()
submission['TARGET'] = final_test_pred
sub_path = f"{OUTPUT_DIR}/submission_v3.csv"
submission.to_csv(sub_path, index=False)
print(f"   Saved: {sub_path}")
print(f"   Shape: {submission.shape}")
print(f"   Prediction range: [{final_test_pred.min():.4f}, {final_test_pred.max():.4f}]")
print(f"\n   First 5 rows:")
print(submission.head().to_string(index=False))

# ─────────────────────────────────────────────
# 12. FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("FINAL SUMMARY — V3 NUCLEAR PIPELINE")
print("=" * 70)
total_time = time.time() - t0
print(f"  LightGBM OOF AUC      : {lgbm_oof_auc:.5f}")
print(f"  CatBoost OOF AUC      : {cat_oof_auc:.5f}")
if USE_XGB:
    print(f"  XGBoost OOF AUC       : {xgb_oof_auc:.5f}")
print(f"  PseudoLabel LGBM OOF  : {roc_auc_score(y, oof_lgbm_final):.5f}")
print(f"  FINAL Ensemble OOF    : {best_auc:.5f}")
print(f"  Total runtime         : {total_time/60:.1f} min")
print(f"  Submission saved      : {sub_path}")
print("=" * 70)
print("READY FOR SUBMISSION — TARGET: BEAT 0.79427")
print("=" * 70)
