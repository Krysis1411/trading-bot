"""
Train the ORB Breakout Quality Scorer.

Reads the labelled trade CSV produced by backtest/rank_symbols.py and trains
a Gradient Boosting classifier to predict whether a breakout will reach the
profit target. Serialises the fitted model to ml/models/breakout_scorer.pkl.

Usage
-----
    /Applications/anaconda3/bin/python -m ml.train_breakout_scorer
"""
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).parent.parent))
from ml.features import FEATURE_COLS, ML_CONFIDENCE_THRESHOLD

DATA_PATH   = Path("backtest/results/breakout_features.csv")
MODEL_DIR   = Path("ml/models")
MODEL_PATH  = MODEL_DIR / "breakout_scorer.pkl"
REPORT_PATH = MODEL_DIR / "breakout_scorer_report.txt"
PLOT_PATH   = MODEL_DIR / "feature_importance.png"


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        sys.exit(
            f"Training data not found at {DATA_PATH}.\n"
            "Run: /Applications/anaconda3/bin/python -m backtest.rank_symbols"
        )
    df = pd.read_csv(DATA_PATH)
    df = df[df["exit_reason"] != "ENGINE_STOP"]   # drop incomplete trades
    df = df.dropna(subset=FEATURE_COLS + ["entry_price", "exit_price"])

    # Label: was the trade profitable? (exit_price > entry_price)
    # This is better than hit_target==TARGET because most wins are EOD closes
    # that still returned positive P&L, not explicit target hits.
    df["hit_target"] = (df["exit_price"] > df["entry_price"]).astype(int)

    total  = len(df)
    wins   = int(df["hit_target"].sum())
    print(f"\nDataset: {total} trades | win rate: {wins/total:.1%}")
    print(f"Exit breakdown:\n{df['exit_reason'].value_counts().to_string()}\n")
    print(f"Symbol breakdown:\n{df['symbol'].value_counts().to_string()}\n")
    return df


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(df: pd.DataFrame):
    X = df[FEATURE_COLS].values
    y = df["hit_target"].values

    # Time-ordered split — last 20% as holdout to avoid look-ahead leakage
    split  = int(len(df) * 0.80)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=42,
    )

    # 5-fold stratified CV on training portion
    cv     = StratifiedKFold(n_splits=5, shuffle=False)
    aucs   = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="roc_auc")
    print(f"CV  ROC-AUC : {aucs.mean():.3f} ± {aucs.std():.3f}")

    # Balanced sample weights so the minority class (hits) gets equal attention
    sw = compute_sample_weight("balanced", y_tr)
    model.fit(X_tr, y_tr, sample_weight=sw)

    # Holdout metrics
    y_prob = model.predict_proba(X_te)[:, 1]
    y_pred = (y_prob >= ML_CONFIDENCE_THRESHOLD).astype(int)
    auc    = roc_auc_score(y_te, y_prob)

    lines = [
        f"Holdout ROC-AUC  : {auc:.3f}",
        f"Holdout samples  : {len(y_te)}",
        f"Confidence gate  : {ML_CONFIDENCE_THRESHOLD:.0%}",
        "",
        classification_report(y_te, y_pred, target_names=["miss", "hit"]),
    ]
    report = "\n".join(lines)
    print(report)

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_te, y_pred, display_labels=["miss", "hit"], ax=ax
    )
    ax.set_title("Breakout Scorer — Holdout Confusion Matrix")
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    return model, report, X_te, y_te


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def plot_importance(model, X_te: np.ndarray, y_te: np.ndarray) -> None:
    result = permutation_importance(
        model, X_te, y_te, n_repeats=20, random_state=42, scoring="roc_auc"
    )
    imp_df = pd.DataFrame({
        "feature":    FEATURE_COLS,
        "importance": result.importances_mean,
        "std":        result.importances_std,
    }).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(
        imp_df["feature"], imp_df["importance"],
        xerr=imp_df["std"], capsize=4, color="steelblue",
    )
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Permutation importance (mean ROC-AUC drop)")
    ax.set_title("ORB Breakout Quality Scorer — Feature Importance")
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()

    print("\nFeature importance (permutation, ROC-AUC):")
    for _, row in imp_df.sort_values("importance", ascending=False).iterrows():
        print(f"  {row['feature']:<22}  {row['importance']:+.4f} ± {row['std']:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    if len(df) < 50:
        print(
            f"Warning: only {len(df)} trades in training data. "
            "Model may be unreliable. Run more backtests first."
        )

    model, report, X_te, y_te = train(df)

    joblib.dump(model, MODEL_PATH)
    REPORT_PATH.write_text(report)
    print(f"\nModel  → {MODEL_PATH}")
    print(f"Report → {REPORT_PATH}")

    plot_importance(model, X_te, y_te)
    print(f"Plots  → {MODEL_DIR}/")


if __name__ == "__main__":
    main()
