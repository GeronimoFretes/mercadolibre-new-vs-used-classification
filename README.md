# Mercado Libre Listing Condition Classification  
#### New vs Used prediction with CatBoost, custom preprocessing, cross-validation, and hyperparameter tuning

This repository contains a machine learning project focused on classifying Mercado Libre listings as **`new`** or **`used`** using a **CatBoost** model trained on tabular and semi-structured product listing data.

The project combines:

- a custom preprocessing and feature engineering pipeline,
- cross-validated model training,
- hyperparameter optimization with **Optuna**,
- frozen preprocessing artifacts for reproducible inference,
- and exported study artifacts for transparency and auditability.

It was developed as an academic machine learning project and is presented here as part of my technical portfolio.

---

## Project overview

Online marketplace listings often contain noisy, incomplete, and semi-structured information. In this project, I approached the problem of classifying listings into **new vs used** by going beyond standard tabular modeling and extracting signal from multiple nested fields such as:

- `attributes`
- `shipping`
- `warranty`
- `variations`
- `tags`
- `pictures`
- `descriptions`

The goal was to build a robust classification pipeline that could transform this raw marketplace data into meaningful features and use them effectively in a high-performance gradient boosting model.

---

## Main components

### 1. Custom preprocessing and feature engineering

A dedicated preprocessing pipeline was built to:

- load raw `.parquet` listing data,
- clean and normalize fields,
- transform nested and semi-structured columns,
- create derived features,
- preserve the final feature schema,
- and ensure consistency between training and inference.

Feature engineering includes signals derived from:

- listing title text
- price bins
- warranty content
- shipping information
- product variations
- attribute combinations
- listing tags
- image-related metadata
- description presence
- category and seller-related information

This makes the project much more than a basic tabular classification task.

---

### 2. Model training with CatBoost

The final classifier is based on **CatBoost**, which is especially well suited for tabular datasets with categorical variables and mixed feature types.

The training setup includes:

- cross-validation,
- fold-level training,
- out-of-fold predictions,
- threshold selection,
- optional calibration support in the training pipeline,
- and artifact export for downstream inference.

---

### 3. Hyperparameter tuning with Optuna

Hyperparameter optimization was performed with **Optuna**, searching over parameters such as:

- tree depth
- learning rate
- regularization strength
- subsampling
- random strength
- feature sampling
- class weighting strategies

The original study was stored in a private database during experimentation, but the relevant public artifacts are now included in this repository so the project can be executed without access to any private infrastructure.

---

## Reproducibility note

This repository now includes the files required to reproduce the **public training and inference workflow**:

- frozen preprocessing pipeline
- best hyperparameters
- Optuna study summary
- Optuna trials export
- original train and test datasets

That said, I do **not** claim exact bit-for-bit reproducibility of the original results across all environments. Small differences may appear depending on:

- package versions,
- operating system,
- execution order,
- CatBoost internal behavior,
- or other environment-specific details.

For that reason, this README intentionally does **not** report a fixed accuracy score as a guaranteed reproducible benchmark from the public repo alone.

The repository should be understood as **fully executable and highly reproducible in workflow**, but not necessarily guaranteed to reproduce the exact same final metric obtained in my original development environment.

---

## Repository structure

```text
mercadolibre-new-vs-used-classification/
├── artifacts/
│   ├── prep_v1/
│   │   ├── pipeline.joblib
│   │   └── schema.json
│   ├── best_params.json
│   ├── study_best.json
│   └── optuna_trials.csv
├── data/
│   ├── train_data.parquet
│   └── test_data.parquet
├── preprocessing/
│   ├── pipeline_io.py
│   └── preprocess_pipeline.py
├── 01-entrenar_guardar_modelo_final.ipynb
├── 02-aplicar_modelo_final_test.ipynb
├── train_catboost_optuna.py
├── LICENSE
└── .gitignore
```

---

## Public artifacts included

To remove the dependency on my private Optuna storage, the repository now includes:

### `artifacts/prep_v1/`

Frozen preprocessing pipeline and schema used to transform the raw dataset consistently.

### `artifacts/best_params.json`

Best hyperparameters used for the final model training.

### `artifacts/study_best.json`

Summary of the best Optuna trial and main study outcome.

### `artifacts/optuna_trials.csv`

Exported trial history from the Optuna study, allowing others to inspect the tuning process.

### `data/train_data.parquet` and `data/test_data.parquet`

Original datasets used for training and inference.

---

## Workflow

### Training workflow

The notebook `01-entrenar_guardar_modelo_final.ipynb` handles the final training process:

1. Load the training dataset
2. Apply the preprocessing pipeline
3. Transform raw listing data into model-ready features
4. Load the best hyperparameters
5. Train CatBoost with cross-validation
6. Save the resulting artifacts for inference

---

### Inference workflow

The notebook `02-aplicar_modelo_final_test.ipynb` applies the saved pipeline and trained model(s) to the test set:

1. Load test data
2. Load the frozen preprocessing pipeline
3. Transform the test set using the same schema as training
4. Generate predictions
5. Export the final submission-ready output

---

## Why this project is interesting

What makes this project especially valuable from a machine learning perspective is that it is not just about fitting a model to clean structured data.

It involves:

* dealing with real-world marketplace data,
* extracting signal from nested fields,
* designing a reusable preprocessing pipeline,
* managing training/inference consistency,
* tuning a strong gradient boosting model,
* and packaging the whole workflow into a repository that others can execute.

This reflects a more realistic applied ML workflow than a simple notebook-based classification experiment.

---

## Tech stack

* **Python**
* **Pandas**
* **NumPy**
* **scikit-learn**
* **CatBoost**
* **Optuna**
* **joblib**

---

## How to run the project

### 1. Clone the repository

```bash
git clone https://github.com/GeronimoFretes/entrega_tp3_ap_2025Q2.git
cd entrega_tp3_ap_2025Q2
```

### 2. Install dependencies

Install the required Python libraries used throughout the project.
A dedicated environment is recommended.

Example:

```bash
pip install pandas numpy scikit-learn catboost optuna joblib pyarrow
```

Depending on your setup, you may also need Jupyter:

```bash
pip install notebook
```

---

### 3. Run training

Open and execute:

```bash
01-entrenar_guardar_modelo_final.ipynb
```

This notebook loads the training data, applies the preprocessing pipeline, reads the saved best hyperparameters, and trains the model.

---

### 4. Run inference

Open and execute:

```bash
02-aplicar_modelo_final_test.ipynb
```

This notebook applies the frozen preprocessing pipeline and the trained model to the test dataset.

---

## Notes on exact replication

The repository is designed so that another user can:

* inspect the full workflow,
* understand the feature engineering logic,
* retrain the model,
* run inference,
* and audit the tuning process.

However, this should not be interpreted as a guarantee of identical final scores under every environment.

For that reason, this repository is best viewed as:

* **fully executable**
* **transparent in methodology**
* **strongly reproducible in process**
* but **not guaranteed to be numerically identical in every run**

---

## Potential future improvements

Some natural next steps for improving the project would be:

* adding a `requirements.txt` or `pyproject.toml`,
* packaging the training pipeline into a CLI or modular training script,
* logging experiments more formally,
* documenting the final feature set in more detail,
* and including a dedicated results section with environment-specific benchmark values.

---

## Author

**Gerónimo Fretes**

Data Science / Machine Learning student at ITBA, with a strong interest in applied ML, structured problem solving, and production-oriented modeling workflows.

This repository is part of my academic and technical portfolio.

---

```

A small recommendation: for portfolio purposes, I would rename the title from the current academic-style repo name inside the README heading, even if the repository URL stays the same. For example:

- **Mercado Libre Listing Condition Classification**
- **New vs Used Classification for Marketplace Listings**
- **Mercado Libre New vs Used Prediction with CatBoost**

The repo can still be called `entrega_tp3_ap_2025Q2`, but the README title should feel more professional and self-explanatory.
```
