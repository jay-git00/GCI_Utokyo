"""
Phase 1: Data Audit
- Column list, dtypes, shapes
- Missing values
- Target distribution
- Key feature previews
"""
import pandas as pd
import numpy as np
import sys

DATA_DIR = r"c:\Users\DILEEP\Downloads\PLACEMENT_PREP\Course_omnicampus\competition_data"

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")
sample_sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

print(f"train shape : {train.shape}")
print(f"test  shape : {test.shape}")
print(f"sample_sub  : {sample_sub.shape}")

print("\n" + "=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)
vc = train['TARGET'].value_counts()
print(vc)
print(f"Default rate: {train['TARGET'].mean()*100:.2f}%")

print("\n" + "=" * 70)
print("COLUMNS IN TRAIN")
print("=" * 70)
for col in sorted(train.columns):
    print(col)

print("\n" + "=" * 70)
print("COLUMNS IN TEST BUT NOT IN TRAIN (excl TARGET)")
print("=" * 70)
train_cols = set(train.columns) - {'TARGET'}
test_cols  = set(test.columns)
only_in_test = test_cols - train_cols
only_in_train = train_cols - test_cols
print(f"Only in test  : {only_in_test}")
print(f"Only in train : {only_in_train}")

print("\n" + "=" * 70)
print("MISSING VALUE SUMMARY (train - top 30)")
print("=" * 70)
miss = train.isnull().sum()
miss_pct = (miss / len(train) * 100).round(2)
miss_df = pd.DataFrame({'missing': miss, 'pct': miss_pct})
miss_df = miss_df[miss_df['missing'] > 0].sort_values('pct', ascending=False)
print(miss_df.head(30).to_string())

print("\n" + "=" * 70)
print("DTYPES SUMMARY")
print("=" * 70)
print(train.dtypes.value_counts())

print("\n" + "=" * 70)
print("NUMERIC FEATURE STATS - KEY COLS")
print("=" * 70)
key_cols = ['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3',
            'AMT_CREDIT','AMT_INCOME_TOTAL','AMT_ANNUITY',
            'AMT_GOODS_PRICE','DAYS_EMPLOYED','DAYS_BIRTH',
            'CNT_FAM_MEMBERS']
existing_key = [c for c in key_cols if c in train.columns]
print(f"Key cols present: {existing_key}")
print(train[existing_key].describe().T.to_string())

print("\nDone. Data audit complete.")
