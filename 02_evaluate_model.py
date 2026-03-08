from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "artifacts"
DEFAULT_OUTPUT_DIR = DEFAULT_ARTIFACTS_DIR / "evaluation_reports"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def pick_existing_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"None of these columns were found: {candidates}")


def compute_report(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(
        y_true,
        y_pred,
        target_names=["used", "new"],
        digits=4,
        output_dict=True,
        zero_division=0,
    )

    return {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
        "classification_report": cr,
        "y_pred": y_pred,
    }


def save_confusion_matrix_csv(cm: np.ndarray, path: Path) -> None:
    cm_df = pd.DataFrame(
        cm,
        index=["true_used", "true_new"],
        columns=["pred_used", "pred_new"],
    )
    cm_df.to_csv(path, index=True)


def run_evaluation(artifacts_dir: Path, output_dir: Path, top_n_features: int) -> None:
    ensure_dir(output_dir)

    oof_path = artifacts_dir / "oof.parquet"
    metrics_path = artifacts_dir / "metrics.json"
    feature_importance_path = artifacts_dir / "feature_importance.csv"
    oof_cal_path = artifacts_dir / "oof_calibrated.parquet"
    inference_config_path = artifacts_dir / "inference_config.json"
    ensemble_info_path = artifacts_dir / "ensemble_info.json"

    if not oof_path.exists():
        raise FileNotFoundError(f"Missing required file: {oof_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing required file: {metrics_path}")

    print("Loading artifacts...")
    oof_df = pd.read_parquet(oof_path)
    metrics = load_json(metrics_path)

    y_col = pick_existing_col(oof_df, ["y", "y_true"])
    prob_col = pick_existing_col(oof_df, ["oof_prob", "prob", "y_prob"])

    y_true = oof_df[y_col].to_numpy().astype(int)
    y_prob = oof_df[prob_col].to_numpy().astype(float)

    base_threshold = float(metrics.get("best_threshold", 0.5))

    print(f"Using OOF probability column: {prob_col}")
    print(f"Using target column: {y_col}")
    print(f"Using selected threshold from metrics.json: {base_threshold:.6f}")

    base_report_best = compute_report(y_true, y_prob, base_threshold)
    base_report_05 = compute_report(y_true, y_prob, 0.5)

    # Save OOF predictions with explicit labels
    oof_eval_df = pd.DataFrame(
        {
            "y_true": y_true,
            "oof_prob": y_prob,
            "pred_05": base_report_05["y_pred"],
            "pred_best_threshold": base_report_best["y_pred"],
        }
    )
    oof_eval_df.to_parquet(output_dir / "oof_predictions_evaluated.parquet", index=False)

    # Save confusion matrices
    save_confusion_matrix_csv(
        np.array(base_report_05["confusion_matrix"]),
        output_dir / "confusion_matrix_threshold_0_5.csv",
    )
    save_confusion_matrix_csv(
        np.array(base_report_best["confusion_matrix"]),
        output_dir / "confusion_matrix_best_threshold.csv",
    )

    # Optional calibrated evaluation
    calibrated_summary: Optional[Dict[str, Any]] = None
    if oof_cal_path.exists():
        print("Found calibrated OOF predictions. Evaluating calibrated probabilities...")

        oof_cal_df = pd.read_parquet(oof_cal_path)
        cal_prob_col = pick_existing_col(oof_cal_df, ["oof_prob_cal", "oof_prob", "prob", "y_prob"])
        y_cal_col = pick_existing_col(oof_cal_df, ["y", "y_true"])

        y_true_cal = oof_cal_df[y_cal_col].to_numpy().astype(int)
        y_prob_cal = oof_cal_df[cal_prob_col].to_numpy().astype(float)

        cal_threshold = float(metrics.get("best_threshold_calibrated", 0.5))

        cal_report_best = compute_report(y_true_cal, y_prob_cal, cal_threshold)
        cal_report_05 = compute_report(y_true_cal, y_prob_cal, 0.5)

        calibrated_summary = {
            "probability_column": cal_prob_col,
            "accuracy_at_0_5": cal_report_05["accuracy"],
            "accuracy_at_best_threshold": cal_report_best["accuracy"],
            "best_threshold": cal_threshold,
            "confusion_matrix_at_0_5": cal_report_05["confusion_matrix"],
            "confusion_matrix_at_best_threshold": cal_report_best["confusion_matrix"],
            "classification_report_at_0_5": cal_report_05["classification_report"],
            "classification_report_at_best_threshold": cal_report_best["classification_report"],
        }

        oof_cal_eval_df = pd.DataFrame(
            {
                "y_true": y_true_cal,
                "oof_prob_cal": y_prob_cal,
                "pred_05": cal_report_05["y_pred"],
                "pred_best_threshold": cal_report_best["y_pred"],
            }
        )
        oof_cal_eval_df.to_parquet(output_dir / "oof_calibrated_predictions_evaluated.parquet", index=False)

        save_confusion_matrix_csv(
            np.array(cal_report_05["confusion_matrix"]),
            output_dir / "confusion_matrix_calibrated_threshold_0_5.csv",
        )
        save_confusion_matrix_csv(
            np.array(cal_report_best["confusion_matrix"]),
            output_dir / "confusion_matrix_calibrated_best_threshold.csv",
        )

    # Optional feature importance summary
    top_features = None
    if feature_importance_path.exists():
        fi = pd.read_csv(feature_importance_path)
        if {"feature", "importance"}.issubset(fi.columns):
            fi_sorted = fi.sort_values("importance", ascending=False).reset_index(drop=True)
            fi_sorted.to_csv(output_dir / "feature_importance_full.csv", index=False)
            fi_sorted.head(top_n_features).to_csv(output_dir / "feature_importance_top.csv", index=False)
            top_features = fi_sorted.head(top_n_features).to_dict(orient="records")

    # Optional extra metadata
    inference_config = load_json(inference_config_path) if inference_config_path.exists() else None
    ensemble_info = load_json(ensemble_info_path) if ensemble_info_path.exists() else None

    summary = {
        "evaluation_type": "artifact_based_oof_evaluation",
        "artifacts_dir": str(artifacts_dir),
        "n_rows": int(len(y_true)),
        "base_oof": {
            "probability_column": prob_col,
            "accuracy_at_0_5": base_report_05["accuracy"],
            "accuracy_at_best_threshold": base_report_best["accuracy"],
            "best_threshold": base_threshold,
            "confusion_matrix_at_0_5": base_report_05["confusion_matrix"],
            "confusion_matrix_at_best_threshold": base_report_best["confusion_matrix"],
            "classification_report_at_0_5": base_report_05["classification_report"],
            "classification_report_at_best_threshold": base_report_best["classification_report"],
        },
        "calibrated_oof": calibrated_summary,
        "metrics_json_snapshot": metrics,
        "ensemble_info": ensemble_info,
        "inference_config_snapshot": inference_config,
        "top_features": top_features,
        "note": (
            "This script does not retrain the model. It evaluates the saved out-of-fold "
            "predictions generated during training, which is the correct non-redundant way "
            "to summarize model performance for this repository."
        ),
    }

    save_json(summary, output_dir / "evaluation_summary.json")

    # Small tabular summary
    rows = [
        {"metric": "Base OOF accuracy @ 0.5", "value": base_report_05["accuracy"]},
        {"metric": "Base OOF best threshold", "value": base_threshold},
        {"metric": "Base OOF accuracy @ best threshold", "value": base_report_best["accuracy"]},
    ]

    if calibrated_summary is not None:
        rows.extend(
            [
                {"metric": "Calibrated OOF accuracy @ 0.5", "value": calibrated_summary["accuracy_at_0_5"]},
                {"metric": "Calibrated OOF best threshold", "value": calibrated_summary["best_threshold"]},
                {"metric": "Calibrated OOF accuracy @ best threshold", "value": calibrated_summary["accuracy_at_best_threshold"]},
            ]
        )

    pd.DataFrame(rows).to_csv(output_dir / "evaluation_overview.csv", index=False)

    print("\nSaved evaluation outputs to:")
    print(f"- {output_dir / 'evaluation_summary.json'}")
    print(f"- {output_dir / 'evaluation_overview.csv'}")
    print(f"- {output_dir / 'oof_predictions_evaluated.parquet'}")
    print(f"- {output_dir / 'confusion_matrix_threshold_0_5.csv'}")
    print(f"- {output_dir / 'confusion_matrix_best_threshold.csv'}")
    if calibrated_summary is not None:
        print(f"- {output_dir / 'oof_calibrated_predictions_evaluated.parquet'}")
        print(f"- {output_dir / 'confusion_matrix_calibrated_threshold_0_5.csv'}")
        print(f"- {output_dir / 'confusion_matrix_calibrated_best_threshold.csv'}")
    if top_features is not None:
        print(f"- {output_dir / 'feature_importance_full.csv'}")
        print(f"- {output_dir / 'feature_importance_top.csv'}")

    print("\nQuick summary:")
    print(f"Base OOF accuracy @ 0.5           : {base_report_05['accuracy']:.6f}")
    print(f"Base OOF best threshold           : {base_threshold:.6f}")
    print(f"Base OOF accuracy @ best threshold: {base_report_best['accuracy']:.6f}")

    if calibrated_summary is not None:
        print(f"Calibrated OOF accuracy @ 0.5           : {calibrated_summary['accuracy_at_0_5']:.6f}")
        print(f"Calibrated OOF best threshold           : {calibrated_summary['best_threshold']:.6f}")
        print(f"Calibrated OOF accuracy @ best threshold: {calibrated_summary['accuracy_at_best_threshold']:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize saved OOF evaluation artifacts without retraining."
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Directory containing oof.parquet, metrics.json, and related artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where evaluation reports will be written.",
    )
    parser.add_argument(
        "--top-n-features",
        type=int,
        default=25,
        help="Number of top features to export in feature_importance_top.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_evaluation(
        artifacts_dir=args.artifacts_dir,
        output_dir=args.output_dir,
        top_n_features=args.top_n_features,
    )


if __name__ == "__main__":
    main()