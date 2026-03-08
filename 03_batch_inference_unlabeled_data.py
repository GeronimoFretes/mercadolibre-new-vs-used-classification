from pathlib import Path

MODELS_DIR = Path("artifacts/models")

required_files = [
    "model_fold0.cbm",
    "model_fold1.cbm",
    "model_fold2.cbm",
    "model_fold3.cbm",
    "model_fold4.cbm",
    "inference_config.json",
]

missing = [f for f in required_files if not (MODELS_DIR / f).exists()]

if missing:
    raise FileNotFoundError(
        "Missing pre-trained model artifacts:\n"
        + "\n".join(missing)
        + "\n\nRun: python scripts/download_models.py"
    )
    

import pandas as pd
import numpy as np

import os
import sys

sys.path.append(os.path.abspath("preprocessing"))

from preprocessing.pipeline_io import load_pipeline

from train_catboost_optuna import make_submission


TEST_DATA_PATH = 'data/test_data.parquet'

PIPELINE_PATH = 'artifacts/prep_v1'

COLS_TO_DROP_EARLY = ['subtitle','differential_pricing','international_delivery_mode','listing_source','site_id', 'coverage_areas']
DATE_COLS = ['date_created','last_updated']
TARGET = "condition"

SUBMISSION_PATH = Path("submission_5fold_catboost_optuna.csv")

df_test = pd.read_parquet(
    TEST_DATA_PATH, 
)
df_test = df_test.drop(columns=COLS_TO_DROP_EARLY).copy()

try:
    pipe_loaded, _ = load_pipeline(PIPELINE_PATH)
    X = pipe_loaded.transform(df_test)
except FileNotFoundError:
    print("Pipeline not found. Please run the training pipeline first.")
    
X['ID'] = range(1, len(X) + 1)

out = make_submission(
    test_df=X,
    id_col='ID',
    model_dir=Path('artifacts/models'),
    submission_path=SUBMISSION_PATH,
)
print(f"Submission saved to: {out.resolve()}")

subm = pd.read_csv(SUBMISSION_PATH)
subm[TARGET] = np.where(subm[TARGET] == 1, 'new', 'used')
subm.to_csv(SUBMISSION_PATH, index=False)
