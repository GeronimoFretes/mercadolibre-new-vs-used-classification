import pandas as pd
from pathlib import Path

import os
import sys

sys.path.append(os.path.abspath("preprocessing"))

import preprocessing.preprocess_pipeline as pp
from preprocessing.pipeline_io import save_pipeline, load_pipeline

artifacts = Path('artifacts')
artifacts.mkdir(parents=True, exist_ok=True)

TRAIN_DATA_PATH = 'data/train_data.parquet'

PIPELINE_PATH = artifacts / 'prep_v1'
USE_LOCAL_PIPELINE = True

COLS_TO_DROP_EARLY = ['subtitle','differential_pricing','international_delivery_mode','listing_source','site_id', 'coverage_areas']
DATE_COLS = ['date_created','last_updated']
TARGET = "condition"

df_train = pd.read_parquet(
    TRAIN_DATA_PATH, 
     
    convert_dates=DATE_COLS
)

df_train = df_train.drop(columns=COLS_TO_DROP_EARLY).copy()

if not USE_LOCAL_PIPELINE:
    pipe = pp.FeaturePipeline().fit(df_train)
    save_pipeline(pipe, out_dir=PIPELINE_PATH)
    
pipe_loaded, _ = load_pipeline(PIPELINE_PATH) 
y = df_train[TARGET]           
df_train_processed = pipe_loaded.transform(df_train)
df_train_processed['condition'] = y.copy()
df_train_processed.to_parquet('data/processed_train_data.parquet', index=False)
