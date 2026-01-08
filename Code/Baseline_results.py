import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
import joblib

# Optional XGBoost
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# --------------------------------------------------
# CONFIG (PORTABLE & USER-SAFE)
# --------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
POS_LABEL = 1

BASE_DIR = Path.home() / "Desktop/New_Dataset"
EMB_PATH = BASE_DIR / "Email_Embeddings.npy"
LBL_PATH = BASE_DIR / "Email_Labels.npy"
OUT_DIR = BASE_DIR / "baseline_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
X = np.load(EMB_PATH)
y = np.load(LBL_PATH)

# --------------------------------------------------
# TRAIN / TEST SPLIT
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

# --------------------------------------------------
# SCALING (LOGISTIC REGRESSION ONLY)
# --------------------------------------------------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

joblib.dump(scaler, OUT_DIR / "scaler.joblib")

# --------------------------------------------------
# PLOTTING HELPERS
# --------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def plot_roc(y_true, y_prob, title, filename):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def plot_pr(y_true, y_prob, title, filename):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

# --------------------------------------------------
# EVALUATION (DIAGRAMS ONLY)
# --------------------------------------------------
def evaluate_with_plots(model, X, y, name):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    plot_confusion_matrix(
        y, y_pred,
        f"{name} – Confusion Matrix",
        OUT_DIR / f"{name}_confusion_matrix.png",
    )

    plot_roc(
        y, y_prob,
        f"{name} – ROC Curve",
        OUT_DIR / f"{name}_roc_curve.png",
    )

    plot_pr(
        y, y_prob,
        f"{name} – Precision–Recall Curve",
        OUT_DIR / f"{name}_pr_curve.png",
    )

    return {
        "Model": name,
        "Accuracy": accuracy_score(y, y_pred),
        "Precision": precision_score(y, y_pred, pos_label=POS_LABEL),
        "Recall": recall_score(y, y_pred, pos_label=POS_LABEL),
        "F1": f1_score(y, y_pred, pos_label=POS_LABEL),
        "ROC_AUC": roc_auc_score(y, y_prob),
    }

# --------------------------------------------------
# 1) LOGISTIC REGRESSION
# --------------------------------------------------
lr = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    solver="lbfgs",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

lr.fit(X_train_s, y_train)
joblib.dump(lr, OUT_DIR / "logistic_regression.joblib")

lr_metrics = evaluate_with_plots(
    lr, X_test_s, y_test, "Logistic_Regression"
)

# --------------------------------------------------
# 2) RANDOM FOREST
# --------------------------------------------------
rf = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

rf.fit(X_train, y_train)
joblib.dump(rf, OUT_DIR / "random_forest.joblib")

rf_metrics = evaluate_with_plots(
    rf, X_test, y_test, "Random_Forest"
)

# --------------------------------------------------
# 3) XGBOOST (OPTIONAL)
# --------------------------------------------------
xgb_metrics = None

if HAS_XGB:
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    xgb_clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )

    xgb_clf.fit(X_train, y_train)
    joblib.dump(xgb_clf, OUT_DIR / "xgboost.joblib")

    xgb_metrics = evaluate_with_plots(
        xgb_clf, X_test, y_test, "XGBoost"
    )

# --------------------------------------------------
# SAVE METRICS SUMMARY
# --------------------------------------------------
results = [lr_metrics, rf_metrics]
if xgb_metrics is not None:
    results.append(xgb_metrics)

results_df = pd.DataFrame(results)
results_df.to_csv(OUT_DIR / "baseline_model_results.csv", index=False)

print("Training complete. Diagrams saved to baseline_results folder.")