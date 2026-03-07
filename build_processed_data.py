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
     
    
)

'''
Traceback (most recent call last):
  File "/home/runner/work/mercadolibre-new-vs-used-classification/mercadolibre-new-vs-used-classification/build_processed_data.py", line 24, in <module>
    df_train = pd.read_parquet(
               ^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.11.14/x64/lib/python3.11/site-packages/pandas/io/parquet.py", line 669, in read_parquet
    return impl.read(
           ^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.11.14/x64/lib/python3.11/site-packages/pandas/io/parquet.py", line 265, in read
    pa_table = self.api.parquet.read_table(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: read_table() got an unexpected keyword argument 'convert_dates'
'''


df_train = df_train.drop(columns=COLS_TO_DROP_EARLY).copy()

if not USE_LOCAL_PIPELINE:
    pipe = pp.FeaturePipeline().fit(df_train)
    save_pipeline(pipe, out_dir=PIPELINE_PATH)
    
pipe_loaded, _ = load_pipeline(PIPELINE_PATH) 
y = df_train[TARGET]           
df_train_processed = pipe_loaded.transform(df_train)
df_train_processed['condition'] = y.copy()
df_train_processed.to_parquet('data/processed_train_data.parquet', index=False)
