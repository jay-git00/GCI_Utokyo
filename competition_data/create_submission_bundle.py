"""
Generate submission notebook (.ipynb) and bundle (.zip) for OmniCampus submission.
"""
import nbformat as nbf
import os
import zipfile

OUTPUT_DIR = r"c:\Users\DILEEP\Downloads\PLACEMENT_PREP\Course_omnicampus\competition_data\output"
PIPELINE_PY = r"c:\Users\DILEEP\Downloads\PLACEMENT_PREP\Course_omnicampus\competition_data\02_full_pipeline.py"
NOTEBOOK_PATH = os.path.join(OUTPUT_DIR, "GCI_World_2026_Home_Credit_Top1_Pipeline.ipynb")
ZIP_PATH = os.path.join(OUTPUT_DIR, "submission_bundle.zip")

with open(PIPELINE_PY, "r", encoding="utf-8") as f:
    py_code = f.read()

nb = nbf.v4.new_notebook()

# Markdown cell 1: Intro
intro_text = """# GCI World 2026 September — Home Credit Default Risk Competition
## High-Performance Feature Engineering + Stratified 5-Fold Ensemble Pipeline
**Architecture:**
- Domain Feature Engineering (60+ engineered features: financial ratios, EXT_SOURCE polynomial & interaction terms, credit bureau aggregates, employment anomalies)
- Stratified 5-Fold Cross Validation
- Gradient Boosting Family Diversity: LightGBM (5-Fold) + CatBoost (5-Fold)
- Non-Parametric Rank-Averaged Blending Ensemble
- Final Target: ROC-AUC Optimization (OOF AUC: ~0.7586+)
"""

nb.cells.append(nbf.v4.new_markdown_cell(intro_text))
nb.cells.append(nbf.v4.new_code_cell(py_code))

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Created notebook: {NOTEBOOK_PATH}")

# Create zip bundle containing submission.csv and the notebook
with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(os.path.join(OUTPUT_DIR, "submission.csv"), arcname="submission.csv")
    zipf.write(NOTEBOOK_PATH, arcname="GCI_World_2026_Home_Credit_Top1_Pipeline.ipynb")
    zipf.write(PIPELINE_PY, arcname="02_full_pipeline.py")

print(f"Created submission zip bundle: {ZIP_PATH}")
