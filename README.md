# Mercado Libre New vs Used Classification

Applied machine learning project for classifying Mercado Libre listings as **`new`** or **`used`** using a **CatBoost** model trained on tabular and semi-structured marketplace data.

This repository is structured as a reusable ML workflow and it includes:

- custom preprocessing and feature engineering
- CatBoost training with cross-validation
- Optuna-based hyperparameter tuning
- saved evaluation artifacts from out-of-fold predictions
- fast inference with pre-trained models distributed through **GitHub Releases**
- an optional GitHub Actions workflow for long-running experiments

---

## Project overview

Marketplace listings combine structured fields with noisy text and nested metadata. In this project, I built a classification pipeline that predicts whether a listing corresponds to a **new** or **used** product by extracting signal from fields such as:

- `attributes`
- `shipping`
- `warranty`
- `variations`
- `tags`
- `pictures`
- `descriptions`

The goal was to build a realistic applied ML workflow: robust preprocessing, strong tabular modeling, careful evaluation, and a clean inference path.

---

## What makes this project interesting

This is more than a basic classifier. The project includes:

- **feature engineering over nested marketplace data**
- **frozen preprocessing for training/inference consistency**
- **cross-validated CatBoost training**
- **Optuna tuning with exported study artifacts**
- **artifact-based evaluation without retraining**
- **release-based distribution of pre-trained models**

---

## Repository structure

```text
mercadolibre-new-vs-used-classification/
├── .github/workflows/optuna.yml
├── artifacts/
│   ├── prep_v1/
│   │   ├── pipeline.joblib
│   │   └── schema.json
│   ├── models/
│   │   └── .gitkeep
│   ├── best_params.json
│   ├── study_best.json
│   ├── optuna_trials.csv
│   ├── oof.parquet
│   └── metrics.json
├── data/
│   ├── train_data.parquet
│   ├── test_data.parquet
├── preprocessing/
│   ├── pipeline_io.py
│   └── preprocess_pipeline.py
├── scripts/
│   ├── download_models.py
├── build_processed_data.py
├── train_catboost_optuna.py
├── 01_train_final_model.ipynb
├── 02_evaluate_model.py
├── 03_batch_inference_unlabeled_data.py
├── requirements.txt
└── README.md
```

---

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/GeronimoFretes/mercadolibre-new-vs-used-classification.git
cd mercadolibre-new-vs-used-classification
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download pre-trained models

```bash
python scripts/download_models.py
```

### 4. Run inference

Open:

```text
03_batch_inference_unlabeled_data.ipynb
```

This is the fastest way to try the project without retraining the full model stack.

---

## Training workflow

To reproduce the public training workflow:

### Build processed training data

```bash
python build_processed_data.py
```

### Train the final model workflow

Open:

```text
01_train_final_model.ipynb
```

This generates the main training artifacts, including:

* out-of-fold predictions
* evaluation metrics
* feature importances
* fold models
* inference configuration

---

## Evaluation workflow

To summarize model performance from saved training artifacts:

```bash
python 02_evaluate_model.py
```

This script does **not** retrain the model. It reads the saved out-of-fold predictions and metrics to produce a clean evaluation summary.

---

## Pre-trained models

The full fold models are **not stored directly in Git history** because of file size constraints.

Instead, they are distributed through **GitHub Releases** and downloaded into:

```text
artifacts/models/
```

This keeps the repository lightweight while still supporting fast inference.

---

## Optuna workflow

The repository also includes an optional automated experimentation workflow:

```text
.github/workflows/optuna.yml
```

This workflow is intended for long-running Optuna studies and is separate from the main public execution path.

It is useful for experimentation, but not required for:

* quick inference
* evaluation
* or the main training workflow

---

## Reproducibility note

This repository is designed to be:

* **fully executable**
* **transparent in methodology**
* **strongly reproducible in workflow**

Exact numeric replication across environments is not guaranteed due to differences in package versions, operating systems, and model execution details.

---

## Tech stack

* Python
* Pandas
* NumPy
* scikit-learn
* CatBoost
* Optuna
* joblib
* GitHub Actions

---

## Author

**Gerónimo Fretes**

Data Science student at ITBA, focused on applied ML, structured problem solving, and production-oriented workflows.
