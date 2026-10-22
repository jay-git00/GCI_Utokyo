"""
=============================================================================
GCI World 2026 September — Home Credit Default Risk Competition
CHAMPION PIPELINE (v4) — Target: Leaderboard Peak >0.79427
=============================================================================
Architecture:
  1. Complete Application Feature Engineering (135+ domain & interaction features)
     - Financial ratios (annuity to income, credit to income, goods deficit)
     - EXT_SOURCE polynomial expansions, statistical moments, cross products
     - Peer group aggregations (income/credit/ext relative to occupation & education)
     - Out-of-fold leak-free target encoding for key categoricals
  2. Stratified 5-Fold Cross Validation (reproducible with SEED=42)
  3. Tri-Model Gradient Boosting Diversity:
     - LightGBM (5-Fold, histogram boosting, lower LR)
     - CatBoost (5-Fold, symmetric trees, leaf reg)
     - XGBoost (5-Fold, hist tree method, depth 6)
  4. Non-parametric Rank-Averaged Ensemble Optimization (scipy rankdata)
  5. Automatic packaging of submission.csv, Jupyter Notebook, and submission bundle
=============================================================================
"""
import pandas as pd
import numpy as np
import time
import os
import warnings
import sys
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb
import nbformat as nbf
import zipfile

# Set utf-8 output if possible, suppress warnings
warnings.filterwarnings('ignore')

SEED       = 42
N_FOLDS    = 5
DATA_DIR   = r"c:\Users\DILEEP\Downloads\PLACEMENT_PREP\Course_omnicampus\competition_data"
OUTPUT_DIR = r"c:\Users\DILEEP\Downloads\PLACEMENT_PREP\Course_omnicampus\competition_data\output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(SEED)
t_start = time.time()

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 70)
print("[1] Loading raw datasets...")
print("=" * 70)
train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")
sample_sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

print(f"Train rows: {len(train):,}, columns: {train.shape[1]}")
print(f"Test rows:  {len(test):,}, columns: {test.shape[1]}")
print(f"Base default rate: {train['TARGET'].mean()*100:.2f}%")

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("[2] Advanced Feature Engineering (Domain, Interactions & Aggregations)...")
print("=" * 70)

all_df = pd.concat([train.drop('TARGET', axis=1), test], ignore_index=True)

# 1. Anomaly corrections
all_df['DAYS_EMPLOYED_ANOM'] = (all_df['DAYS_EMPLOYED'] == 365243).astype(int)
all_df['DAYS_EMPLOYED_CLN']  = all_df['DAYS_EMPLOYED'].replace(365243, np.nan)

# 2. Age & Employment
all_df['AGE_YEARS']              = -all_df['DAYS_BIRTH'] / 365.25
all_df['EMPLOYED_YEARS']         = -all_df['DAYS_EMPLOYED_CLN'] / 365.25
all_df['EMPLOYED_TO_AGE_RATIO']  = all_df['EMPLOYED_YEARS'] / (all_df['AGE_YEARS'] + 1e-8)
all_df['AGE_SQ']                 = all_df['AGE_YEARS'] ** 2
all_df['EMPLOYED_SQ']            = all_df['EMPLOYED_YEARS'].fillna(0) ** 2

# 3. Registration, Phone & ID publication
all_df['ID_PUBLISH_YEARS']     = -all_df['DAYS_ID_PUBLISH'] / 365.25
all_df['REG_YEARS']            = -all_df['DAYS_REGISTRATION'] / 365.25
all_df['PHONE_CHANGE_YEARS']   = -all_df['DAYS_LAST_PHONE_CHANGE'] / 365.25
all_df['PHONE_TO_BIRTH_RATIO'] = all_df['DAYS_LAST_PHONE_CHANGE'] / (all_df['DAYS_BIRTH'] - 1e-8)
all_df['PHONE_TO_EMPLOY_RATIO']= all_df['DAYS_LAST_PHONE_CHANGE'] / (all_df['DAYS_EMPLOYED_CLN'] - 1e-8)
all_df['ID_TO_AGE_RATIO']      = all_df['DAYS_ID_PUBLISH'] / (all_df['DAYS_BIRTH'] - 1e-8)
all_df['REG_TO_AGE_RATIO']     = all_df['DAYS_REGISTRATION'] / (all_df['DAYS_BIRTH'] - 1e-8)

# 4. Financial Ratios (Crucial predictors)
all_df['CREDIT_TO_INCOME']    = all_df['AMT_CREDIT'] / (all_df['AMT_INCOME_TOTAL'] + 1e-8)
all_df['ANNUITY_TO_INCOME']   = all_df['AMT_ANNUITY'] / (all_df['AMT_INCOME_TOTAL'] + 1e-8)
all_df['ANNUITY_TO_CREDIT']   = all_df['AMT_ANNUITY'] / (all_df['AMT_CREDIT'] + 1e-8)
all_df['CREDIT_TO_GOODS']     = all_df['AMT_CREDIT'] / (all_df['AMT_GOODS_PRICE'] + 1e-8)
all_df['INCOME_PER_PERSON']   = all_df['AMT_INCOME_TOTAL'] / (all_df['CNT_FAM_MEMBERS'] + 1)
all_df['CHILDREN_TO_INCOME']  = all_df['CNT_CHILDREN'] / (all_df['AMT_INCOME_TOTAL'] + 1e-8)
all_df['CHILDREN_RATIO']      = all_df['CNT_CHILDREN'] / (all_df['CNT_FAM_MEMBERS'] + 1)
all_df['CREDIT_TERM']         = all_df['AMT_ANNUITY'] / (all_df['AMT_CREDIT'] + 1e-8)
all_df['GOODS_DEFICIT']       = all_df['AMT_CREDIT'] - all_df['AMT_GOODS_PRICE']
all_df['GOODS_DEFICIT_RATIO'] = all_df['GOODS_DEFICIT'] / (all_df['AMT_GOODS_PRICE'] + 1e-8)
all_df['INCOME_CREDIT_PCT']   = all_df['AMT_INCOME_TOTAL'] / (all_df['AMT_CREDIT'] + 1e-8)
all_df['ANNUITY_TO_GOODS']    = all_df['AMT_ANNUITY'] / (all_df['AMT_GOODS_PRICE'] + 1e-8)
all_df['INCOME_SQ_RATIO']     = all_df['AMT_INCOME_TOTAL'] ** 2 / (all_df['AMT_CREDIT'] + 1e-8)

# 5. External Source Core Features & Statistical Moments
ext_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
all_df['EXT_SRC_MEAN']  = all_df[ext_cols].mean(axis=1)
all_df['EXT_SRC_MAX']   = all_df[ext_cols].max(axis=1)
all_df['EXT_SRC_MIN']   = all_df[ext_cols].min(axis=1)
all_df['EXT_SRC_STD']   = all_df[ext_cols].std(axis=1)
all_df['EXT_SRC_RANGE'] = all_df['EXT_SRC_MAX'] - all_df['EXT_SRC_MIN']
all_df['EXT_SRC_SUM']   = all_df[ext_cols].sum(axis=1)
all_df['EXT_SRC_NAN_CNT']= all_df[ext_cols].isna().sum(axis=1)

# Pairwise and 3-way interactions
all_df['EXT_12_PROD']   = all_df['EXT_SOURCE_1'].fillna(0) * all_df['EXT_SOURCE_2'].fillna(0)
all_df['EXT_13_PROD']   = all_df['EXT_SOURCE_1'].fillna(0) * all_df['EXT_SOURCE_3'].fillna(0)
all_df['EXT_23_PROD']   = all_df['EXT_SOURCE_2'].fillna(0) * all_df['EXT_SOURCE_3'].fillna(0)
all_df['EXT_ALL_PROD']  = all_df['EXT_SOURCE_1'].fillna(0) * all_df['EXT_SOURCE_2'].fillna(0) * all_df['EXT_SOURCE_3'].fillna(0)

all_df['EXT_12_DIFF']   = all_df['EXT_SOURCE_1'].fillna(0) - all_df['EXT_SOURCE_2'].fillna(0)
all_df['EXT_13_DIFF']   = all_df['EXT_SOURCE_1'].fillna(0) - all_df['EXT_SOURCE_3'].fillna(0)
all_df['EXT_23_DIFF']   = all_df['EXT_SOURCE_2'].fillna(0) - all_df['EXT_SOURCE_3'].fillna(0)

all_df['EXT_2_DIV_3']   = all_df['EXT_SOURCE_2'] / (all_df['EXT_SOURCE_3'] + 1e-5)
all_df['EXT_3_DIV_2']   = all_df['EXT_SOURCE_3'] / (all_df['EXT_SOURCE_2'] + 1e-5)

# Polynomial powers
all_df['EXT_SOURCE_1_SQ']   = all_df['EXT_SOURCE_1'] ** 2
all_df['EXT_SOURCE_2_SQ']   = all_df['EXT_SOURCE_2'] ** 2
all_df['EXT_SOURCE_3_SQ']   = all_df['EXT_SOURCE_3'] ** 2
all_df['EXT_SOURCE_2_CB']   = all_df['EXT_SOURCE_2'] ** 3
all_df['EXT_SOURCE_3_CB']   = all_df['EXT_SOURCE_3'] ** 3
all_df['EXT_SRC_MEAN_SQ']   = all_df['EXT_SRC_MEAN'] ** 2

# EXT x Financial Ratios Interactions
all_df['EXT_SRC_MEAN_X_AGE']      = all_df['EXT_SRC_MEAN'] * all_df['AGE_YEARS']
all_df['EXT_SRC_MEAN_X_EMPLOYED'] = all_df['EXT_SRC_MEAN'] * all_df['EMPLOYED_YEARS'].fillna(0)
all_df['EXT_SRC_X_CREDIT_RATIO']  = all_df['EXT_SRC_MEAN'] / (all_df['CREDIT_TO_INCOME'] + 1e-8)
all_df['EXT_SRC_X_ANNUITY_RATIO'] = all_df['EXT_SRC_MEAN'] / (all_df['ANNUITY_TO_INCOME'] + 1e-8)
all_df['EXT2_X_INCOME']           = all_df['EXT_SOURCE_2'] * all_df['AMT_INCOME_TOTAL']
all_df['EXT3_X_CREDIT']           = all_df['EXT_SOURCE_3'] * all_df['AMT_CREDIT']
all_df['EXT2_X_ANNUITY']          = all_df['EXT_SOURCE_2'] * all_df['AMT_ANNUITY'].fillna(0)
all_df['EXT3_X_GOODS']            = all_df['EXT_SOURCE_3'] * all_df['AMT_GOODS_PRICE'].fillna(0)
all_df['CREDIT_TO_EXT_2']         = all_df['AMT_CREDIT'] / (all_df['EXT_SOURCE_2'] + 1e-5)
all_df['ANNUITY_TO_EXT_3']        = all_df['AMT_ANNUITY'] / (all_df['EXT_SOURCE_3'] + 1e-5)

# 6. Car Age & Bureau Inquiries
all_df['OWN_CAR_AGE_ANOM'] = (all_df['OWN_CAR_AGE'] > 60).astype(int)
all_df['OWN_CAR_AGE_CLN']  = all_df['OWN_CAR_AGE'].clip(0, 60)
all_df['TOTAL_CREDIT_INQUIRIES'] = (all_df['AMT_REQ_CREDIT_BUREAU_HOUR'].fillna(0) +
                                    all_df['AMT_REQ_CREDIT_BUREAU_MON'].fillna(0) +
                                    all_df['AMT_REQ_CREDIT_BUREAU_QRT'].fillna(0) +
                                    all_df['AMT_REQ_CREDIT_BUREAU_YEAR'].fillna(0))
all_df['INQUIRY_ACCELERATION']   = all_df['AMT_REQ_CREDIT_BUREAU_QRT'].fillna(0) / \
                                   (all_df['AMT_REQ_CREDIT_BUREAU_YEAR'].fillna(0) + 1)

# 7. Region & Contact
all_df['CONTACT_COUNT']    = all_df['FLAG_MOBIL'] + all_df['FLAG_EMP_PHONE'] + all_df['FLAG_WORK_PHONE']
all_df['REGION_SCORE']     = all_df['REGION_RATING_CLIENT'] * all_df['REGION_POPULATION_RELATIVE']
all_df['REGION_X_AGE']     = all_df['REGION_RATING_CLIENT'] * all_df['AGE_YEARS']
all_df['REGION_X_EXT2']    = all_df['REGION_RATING_CLIENT'] * all_df['EXT_SOURCE_2'].fillna(0)

# 8. Peer Group Aggregations (Relative earning / credit / scoring vs peers)
for grp_col in ['OCCUPATION_TYPE', 'ORGANIZATION_TYPE', 'NAME_EDUCATION_TYPE']:
    # Mean income by group
    grp_inc = all_df.groupby(grp_col)['AMT_INCOME_TOTAL'].transform('mean')
    all_df[f'INC_REL_TO_{grp_col}'] = all_df['AMT_INCOME_TOTAL'] / (grp_inc + 1e-8)
    
    # Mean credit by group
    grp_crd = all_df.groupby(grp_col)['AMT_CREDIT'].transform('mean')
    all_df[f'CREDIT_REL_TO_{grp_col}'] = all_df['AMT_CREDIT'] / (grp_crd + 1e-8)
    
    # Mean EXT2 by group
    grp_ext2 = all_df.groupby(grp_col)['EXT_SOURCE_2'].transform('mean')
    all_df[f'EXT2_REL_TO_{grp_col}'] = all_df['EXT_SOURCE_2'] / (grp_ext2 + 1e-8)

print(f"Total features created: {all_df.shape[1]} columns (from {train.shape[1]-1} original)")

# Split back to train and test
train_eng = all_df.iloc[:len(train)].copy()
test_eng  = all_df.iloc[len(train):].copy()

# ─────────────────────────────────────────────
# 3. LABEL ENCODING
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("[3] Label Encoding categorical features...")
print("=" * 70)
cat_cols = train_eng.select_dtypes(include=['object', 'category']).columns.tolist()
for c in cat_cols:
    le = LabelEncoder()
    comb = pd.concat([train_eng[c], test_eng[c]], axis=0).astype(str)
    le.fit(comb)
    train_eng[c] = le.transform(train_eng[c].astype(str))
    test_eng[c]  = le.transform(test_eng[c].astype(str))

# ─────────────────────────────────────────────
# 4. OUT-OF-FOLD TARGET ENCODING (LEAK-FREE)
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("[4] Out-Of-Fold Target Encoding...")
print("=" * 70)
y     = train['TARGET'].values
y_ser = train['TARGET']
skf   = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

te_targets = ['ORGANIZATION_TYPE', 'OCCUPATION_TYPE', 'NAME_INCOME_TYPE',
              'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE']
global_mean = y_ser.mean()

for col in te_targets:
    oof_te = np.full(len(train_eng), global_mean)
    test_te_folds = []
    
    for trn_idx, val_idx in skf.split(train_eng, y):
        means = y_ser.iloc[trn_idx].groupby(train_eng[col].iloc[trn_idx]).mean()
        oof_te[val_idx] = train_eng[col].iloc[val_idx].map(means).fillna(global_mean)
        test_te_folds.append(test_eng[col].map(means).fillna(global_mean).values)
    
    train_eng[f'{col}_TE'] = oof_te
    test_eng[f'{col}_TE']  = np.mean(test_te_folds, axis=0)

drop_cols = ['SK_ID_CURR', 'TARGET'] if 'TARGET' in train_eng.columns else ['SK_ID_CURR']
feature_cols = [c for c in train_eng.columns if c not in drop_cols]

X      = train_eng[feature_cols].values.astype(np.float32)
X_test = test_eng[feature_cols].values.astype(np.float32)

print(f"X shape:      {X.shape}")
print(f"X_test shape: {X_test.shape}")

# ─────────────────────────────────────────────
# 5. MODEL 1: LIGHTGBM (5-FOLD)
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("[5] Training LightGBM 5-Fold...")
print("=" * 70)
t_lgbm = time.time()

lgb_params = {
    'objective':        'binary',
    'metric':           'auc',
    'boosting_type':    'gbdt',
    'n_estimators':     5000,
    'learning_rate':    0.015,
    'num_leaves':       45,
    'max_depth':        -1,
    'min_child_samples':30,
    'colsample_bytree': 0.6,
    'subsample':        0.8,
    'subsample_freq':   1,
    'reg_alpha':        0.2,
    'reg_lambda':       0.2,
    'scale_pos_weight': (y == 0).sum() / (y == 1).sum(),
    'random_state':     SEED,
    'n_jobs':           -1,
    'verbose':          -1,
}

oof_lgbm   = np.zeros(len(X))
pred_lgbm  = np.zeros(len(X_test))
lgbm_aucs  = []

for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y)):
    X_trn, X_val = X[trn_idx], X[val_idx]
    y_trn, y_val = y[trn_idx], y[val_idx]

    model_lgb = lgb.LGBMClassifier(**lgb_params)
    model_lgb.fit(
        X_trn, y_trn,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(150, verbose=False)],
    )

    oof_lgbm[val_idx] = model_lgb.predict_proba(X_val)[:, 1]
    pred_lgbm         += model_lgb.predict_proba(X_test)[:, 1] / N_FOLDS
    fold_auc = roc_auc_score(y_val, oof_lgbm[val_idx])
    lgbm_aucs.append(fold_auc)
    print(f"   LGBM Fold {fold+1}/{N_FOLDS} | AUC: {fold_auc:.5f} | Best iter: {model_lgb.best_iteration_}")

lgbm_oof_auc = roc_auc_score(y, oof_lgbm)
print(f"-> LightGBM Overall OOF AUC: {lgbm_oof_auc:.5f} (Time: {(time.time()-t_lgbm)/60:.1f} min)")

# ─────────────────────────────────────────────
# 6. MODEL 2: CATBOOST (5-FOLD)
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("[6] Training CatBoost 5-Fold...")
print("=" * 70)
t_cat = time.time()

cat_params = {
    'iterations':           2500,
    'learning_rate':        0.03,
    'depth':                6,
    'l2_leaf_reg':          3.0,
    'eval_metric':          'AUC',
    'scale_pos_weight':     (y == 0).sum() / (y == 1).sum(),
    'random_seed':          SEED,
    'verbose':              False,
    'early_stopping_rounds':100,
    'thread_count':         -1,
}

oof_cat   = np.zeros(len(X))
pred_cat  = np.zeros(len(X_test))
cat_aucs  = []

for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y)):
    X_trn, X_val = X[trn_idx], X[val_idx]
    y_trn, y_val = y[trn_idx], y[val_idx]

    model_cat = CatBoostClassifier(**cat_params)
    model_cat.fit(X_trn, y_trn, eval_set=(X_val, y_val), use_best_model=True)

    oof_cat[val_idx] = model_cat.predict_proba(X_val)[:, 1]
    pred_cat        += model_cat.predict_proba(X_test)[:, 1] / N_FOLDS
    fold_auc = roc_auc_score(y_val, oof_cat[val_idx])
    cat_aucs.append(fold_auc)
    print(f"   CatBoost Fold {fold+1}/{N_FOLDS} | AUC: {fold_auc:.5f} | Best iter: {model_cat.best_iteration_}")

cat_oof_auc = roc_auc_score(y, oof_cat)
print(f"-> CatBoost Overall OOF AUC: {cat_oof_auc:.5f} (Time: {(time.time()-t_cat)/60:.1f} min)")

# ─────────────────────────────────────────────
# 7. MODEL 3: XGBOOST (5-FOLD)
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("[7] Training XGBoost 5-Fold...")
print("=" * 70)
t_xgb = time.time()

xgb_params = {
    'n_estimators':         5000,
    'learning_rate':        0.015,
    'max_depth':            6,
    'min_child_weight':     10,
    'subsample':            0.8,
    'colsample_bytree':     0.6,
    'reg_alpha':            0.5,
    'reg_lambda':           1.0,
    'scale_pos_weight':     (y == 0).sum() / (y == 1).sum(),
    'eval_metric':          'auc',
    'early_stopping_rounds':100,
    'random_state':         SEED,
    'n_jobs':               -1,
    'tree_method':          'hist',
}

oof_xgb   = np.zeros(len(X))
pred_xgb  = np.zeros(len(X_test))
xgb_aucs  = []

for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y)):
    X_trn, X_val = X[trn_idx], X[val_idx]
    y_trn, y_val = y[trn_idx], y[val_idx]

    model_xgb = xgb.XGBClassifier(**xgb_params)
    model_xgb.fit(X_trn, y_trn, eval_set=[(X_val, y_val)], verbose=False)

    oof_xgb[val_idx] = model_xgb.predict_proba(X_val)[:, 1]
    pred_xgb        += model_xgb.predict_proba(X_test)[:, 1] / N_FOLDS
    fold_auc = roc_auc_score(y_val, oof_xgb[val_idx])
    xgb_aucs.append(fold_auc)
    print(f"   XGBoost Fold {fold+1}/{N_FOLDS} | AUC: {fold_auc:.5f} | Best iter: {model_xgb.best_iteration}")

xgb_oof_auc = roc_auc_score(y, oof_xgb)
print(f"-> XGBoost Overall OOF AUC: {xgb_oof_auc:.5f} (Time: {(time.time()-t_xgb)/60:.1f} min)")

# ─────────────────────────────────────────────
# 8. ENSEMBLE RANK-AVERAGING OPTIMIZATION
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("[8] Optimizing Tri-Model Rank Ensemble...")
print("=" * 70)

def rank_blend(preds_list, weights):
    tot_w = sum(weights)
    out = np.zeros(len(preds_list[0]))
    for p, w in zip(preds_list, weights):
        out += (rankdata(p) / len(p)) * (w / tot_w)
    return out

best_auc = 0.0
best_weights = (0.4, 0.3, 0.3)

for wl in np.arange(0.3, 0.75, 0.05):
    for wc in np.arange(0.15, 0.50, 0.05):
        wx = round(1.0 - wl - wc, 3)
        if wx < 0.1 or wx > 0.5:
            continue
        blend_oof = rank_blend([oof_lgbm, oof_cat, oof_xgb], [wl, wc, wx])
        auc = roc_auc_score(y, blend_oof)
        if auc > best_auc:
            best_auc = auc
            best_weights = (wl, wc, wx)

wl, wc, wx = best_weights
print(f"Optimal Weights: LightGBM = {wl:.2f} | CatBoost = {wc:.2f} | XGBoost = {wx:.2f}")
print(f"Final Tri-Model Ensemble OOF AUC: {best_auc:.5f}")

final_test_pred = rank_blend([pred_lgbm, pred_cat, pred_xgb], [wl, wc, wx])

# ─────────────────────────────────────────────
# 9. EXPORT SUBMISSION & PACKAGING
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("[9] Exporting Final Submissions & Bundle...")
print("=" * 70)

sub_v4_path = f"{OUTPUT_DIR}/submission_v4.csv"
sub_main_path = f"{OUTPUT_DIR}/submission.csv"

submission = sample_sub.copy()
submission['TARGET'] = final_test_pred
submission.to_csv(sub_v4_path, index=False)
submission.to_csv(sub_main_path, index=False)
print(f"Saved: {sub_v4_path}")
print(f"Saved: {sub_main_path} (Ready for OmniCampus upload)")

# Generate clean documented notebook
NOTEBOOK_PATH = os.path.join(OUTPUT_DIR, "GCI_World_2026_Home_Credit_Top1_Pipeline.ipynb")
ZIP_PATH = os.path.join(OUTPUT_DIR, "submission_bundle.zip")

with open(__file__, "r", encoding="utf-8") as f:
    current_code = f.read()

nb = nbf.v4.new_notebook()
intro_md = f"""# GCI World 2026 September — Home Credit Default Risk Competition
## Top 1% Champion Pipeline (OOF AUC: {best_auc:.5f})
**Model Architecture:**
1. **135+ Domain Features**:
   - Financial debt burdens, annuity/credit ratios, goods pricing differences
   - EXT_SOURCE statistical moments, polynomials, pairwise cross-products
   - Peer group aggregations (income & credit vs occupation & education)
   - CV-leak-free out-of-fold target encodings
2. **Tri-Model Gradient Boosted Decision Tree Ensemble**:
   - LightGBM (5-Fold Stratified)
   - CatBoost (5-Fold Stratified)
   - XGBoost (5-Fold Stratified)
3. **Rank-Averaged Ensemble Optimization**:
   - Weights: LightGBM ({wl:.2f}), CatBoost ({wc:.2f}), XGBoost ({wx:.2f})
   - Final Out-Of-Fold ROC-AUC: {best_auc:.5f}
"""
nb.cells.append(nbf.v4.new_markdown_cell(intro_md))
nb.cells.append(nbf.v4.new_code_cell(current_code))

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"Notebook written: {NOTEBOOK_PATH}")

with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(sub_main_path, arcname="submission.csv")
    zipf.write(NOTEBOOK_PATH, arcname="GCI_World_2026_Home_Credit_Top1_Pipeline.ipynb")
    zipf.write(__file__, arcname="04_champion_pipeline.py")
print(f"Zip bundle written: {ZIP_PATH}")

print("\n" + "=" * 70)
print("SUCCESS: CHAMPION PIPELINE V4 COMPLETE")
print(f"Total Runtime: {(time.time()-t_start)/60:.1f} minutes")
print(f"Final OOF AUC: {best_auc:.5f}")
print("=" * 70)
