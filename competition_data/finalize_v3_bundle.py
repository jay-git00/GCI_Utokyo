"""
Bundle helper to prepare final submission files once V3 finishes.
"""
import os
import shutil
import zipfile
import nbformat as nbf

OUTPUT_DIR = r"c:\Users\DILEEP\Downloads\PLACEMENT_PREP\Course_omnicampus\competition_data\output"
PIPELINE_V3 = r"c:\Users\DILEEP\Downloads\PLACEMENT_PREP\Course_omnicampus\competition_data\03_nuclear_pipeline.py"
SUB_V3 = os.path.join(OUTPUT_DIR, "submission_v3.csv")
SUB_MAIN = os.path.join(OUTPUT_DIR, "submission.csv")
NOTEBOOK_PATH = os.path.join(OUTPUT_DIR, "GCI_World_2026_Home_Credit_Top1_Pipeline.ipynb")
ZIP_PATH = os.path.join(OUTPUT_DIR, "submission_bundle.zip")

def finalize():
    if not os.path.exists(SUB_V3):
        print("submission_v3.csv does not exist yet.")
        return False
    
    # Copy submission_v3.csv -> submission.csv
    shutil.copyfile(SUB_V3, SUB_MAIN)
    print(f"Copied {SUB_V3} -> {SUB_MAIN}")

    # Generate updated notebook with v3 code
    with open(PIPELINE_V3, "r", encoding="utf-8") as f:
        py_code = f.read()

    nb = nbf.v4.new_notebook()
    intro_text = """# GCI World 2026 September — Home Credit Default Risk Competition
## V3 Nuclear Pipeline — Target: Beat 0.79427 (Top 1%)
**Architecture Highlights:**
- 120+ Engineered domain features (financial ratios, EXT_SOURCE polynomial & interaction crosses, leak-free target encodings)
- Stratified 5-Fold Cross Validation
- Tri-model diversity: LightGBM (depth/leaves expanded, 10k trees with early stopping) + CatBoost + XGBoost
- Semi-Supervised Pseudo-Labeling on test high-confidence extremes
- Non-parametric Rank-Averaged Ensemble optimization
"""
    nb.cells.append(nbf.v4.new_markdown_cell(intro_text))
    nb.cells.append(nbf.v4.new_code_cell(py_code))

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Updated notebook: {NOTEBOOK_PATH}")

    # Create zip bundle
    with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(SUB_MAIN, arcname="submission.csv")
        zipf.write(NOTEBOOK_PATH, arcname="GCI_World_2026_Home_Credit_Top1_Pipeline.ipynb")
        zipf.write(PIPELINE_V3, arcname="03_nuclear_pipeline.py")
    print(f"Updated submission bundle: {ZIP_PATH}")
    return True

if __name__ == "__main__":
    finalize()
