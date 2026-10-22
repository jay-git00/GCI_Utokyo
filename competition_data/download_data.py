"""
Download competition data files from Google Drive.
IDs extracted from the folder listing.
"""
import os, urllib.request

OUTPUT_DIR = r"c:\Users\DILEEP\Downloads\PLACEMENT_PREP\Course_omnicampus\competition_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FILES = {
    "train.csv": "1ov4BQJcmSidP_YO5gaszsbfkmQgcRj1F",
    "test.csv":  "1p6qAIGmpPgbfYpm7Clh6cIPm3768RHL0",
    "sample_submission.csv": "1LurryHvRs5rtVFvWVlaj6wqJe4jvw76K",
}

BASE_URL = "https://drive.google.com/uc?export=download&id={}"

for fname, file_id in FILES.items():
    dest = os.path.join(OUTPUT_DIR, fname)
    if os.path.exists(dest):
        size = os.path.getsize(dest)
        print(f"[SKIP] {fname} already exists ({size:,} bytes)")
        continue
    url = BASE_URL.format(file_id)
    print(f"Downloading {fname} ...")
    urllib.request.urlretrieve(url, dest)
    size = os.path.getsize(dest)
    print(f"  -> Saved: {size:,} bytes")

print("Done.")
