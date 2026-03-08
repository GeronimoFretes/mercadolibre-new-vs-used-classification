from pathlib import Path
from urllib.request import urlretrieve

REPO_OWNER = "GeronimoFretes"
REPO_NAME = "mercadolibre-new-vs-used-classification"

ASSETS = {
    "model_fold1.cbm": f"https://github.com/{REPO_OWNER}/{REPO_NAME}/releases/latest/download/model_fold0.cbm",
    "model_fold2.cbm": f"https://github.com/{REPO_OWNER}/{REPO_NAME}/releases/latest/download/model_fold1.cbm",
    "model_fold3.cbm": f"https://github.com/{REPO_OWNER}/{REPO_NAME}/releases/latest/download/model_fold2.cbm",
    "model_fold4.cbm": f"https://github.com/{REPO_OWNER}/{REPO_NAME}/releases/latest/download/model_fold3.cbm",
    "model_fold5.cbm": f"https://github.com/{REPO_OWNER}/{REPO_NAME}/releases/latest/download/model_fold4.cbm",
    "inference_config.json": f"https://github.com/{REPO_OWNER}/{REPO_NAME}/releases/latest/download/inference_config.json",
}

OUTPUT_DIR = Path("artifacts/models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for filename, url in ASSETS.items():
    out_path = OUTPUT_DIR / filename
    if out_path.exists():
        print(f"Skipping {filename} (already exists)")
        continue

    print(f"Downloading {filename}...")
    urlretrieve(url, out_path)

print("Done.")