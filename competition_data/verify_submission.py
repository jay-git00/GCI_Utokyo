import pandas as pd
import numpy as np

sub = pd.read_csv(r"c:\Users\DILEEP\Downloads\PLACEMENT_PREP\Course_omnicampus\competition_data\output\submission.csv")
sample = pd.read_csv(r"c:\Users\DILEEP\Downloads\PLACEMENT_PREP\Course_omnicampus\competition_data\sample_submission.csv")
test = pd.read_csv(r"c:\Users\DILEEP\Downloads\PLACEMENT_PREP\Course_omnicampus\competition_data\test.csv")

checks = {}
checks["1. Row Count Match (61,500)"] = (len(sub) == len(sample) == len(test) == 61500)
checks["2. Column Names Match ('SK_ID_CURR', 'TARGET')"] = (list(sub.columns) == ["SK_ID_CURR", "TARGET"])
checks["3. ID Order 100% Identical to Sample"] = bool((sub["SK_ID_CURR"].values == sample["SK_ID_CURR"].values).all())
checks["4. Zero NaNs or Infs"] = bool(sub.isna().sum().sum() == 0 and not np.isinf(sub["TARGET"]).any())
checks["5. Probability Range [0.0, 1.0]"] = bool(sub["TARGET"].min() >= 0.0 and sub["TARGET"].max() <= 1.0)
checks["6. ID Bounds (171202 to 232701)"] = bool(sub["SK_ID_CURR"].min() == 171202 and sub["SK_ID_CURR"].max() == 232701)

with open(r"c:\Users\DILEEP\Downloads\PLACEMENT_PREP\Course_omnicampus\competition_data\output\submission.csv") as f:
    lines = sum(1 for _ in f)
checks["7. Total File Lines (Header + 61500 = 61501)"] = (lines == 61501)

print("=" * 60)
print("AUDIT REPORT: submission.csv")
print("=" * 60)
all_pass = True
for k, v in checks.items():
    status = "PASS" if v else "FAIL"
    if not v:
        all_pass = False
    print(f"[{status}] {k}")

print("=" * 60)
print(f"Overall Result: {'100% VERIFIED READY' if all_pass else 'ISSUES FOUND'}")
print("=" * 60)
print("Statistics:")
print("  Min :", round(sub['TARGET'].min(), 6))
print("  Max :", round(sub['TARGET'].max(), 6))
print("  Mean:", round(sub['TARGET'].mean(), 6))
print("  Std :", round(sub['TARGET'].std(), 6))
print("\nFirst 5 rows:")
print(sub.head(5).to_string(index=False))
print("\nLast 5 rows:")
print(sub.tail(5).to_string(index=False))
