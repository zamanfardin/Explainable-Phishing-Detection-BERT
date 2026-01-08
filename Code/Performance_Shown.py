import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
import joblib

# Optional XGBoost
try:
    import xgboost as xgb
    HAS_XGB = True
except:
    HAS_XGB = False

# -----------------------------
# CONFIG
# -----------------------------
EMB_PATH = "/Users/fardinzaman/Desktop/Email_Embeddings.npy"
LBL_PATH = "/Users/fardinzaman/Desktop/Email_Labels.npy"
OUT_DIR = "/Users/fardinzaman/Desktop/baseline_results"
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2

# -----------------------------
# LOAD DATA
# -----------------------------
X = np.load(EMB_PATH)
y = np.load(LBL_PATH)

print("Loaded embeddings:", X.shape)
print("Loaded labels:", y.shape)

# -----------------------------
# TRAIN / TEST SPLIT (STRATIFIED)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# -----------------------------
# SCALE (for Logistic Regression)
# -----------------------------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.joblib"))

# -----------------------------
# HELPER FUNCTION
# -----------------------------
def evaluate(model, X, y, name):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y, y_pred),
        "Precision": precision_score(y, y_pred),
        "Recall": recall_score(y, y_pred),
        "F1": f1_score(y, y_pred),
        "ROC_AUC": roc_auc_score(y, y_prob) if y_prob is not None else None
    }

    print(f"\n{name}")
    print(classification_report(y, y_pred))

    cm = confusion_matrix(y, y_pred)
    pd.DataFrame(cm).to_csv(os.path.join(OUT_DIR, f"{name}_confusion_matrix.csv"))

    return metrics

# -----------------------------
# LOGISTIC REGRESSION
# -----------------------------
lr = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=RANDOM_STATE
)
lr.fit(X_train_s, y_train)

joblib.dump(lr, os.path.join(OUT_DIR, "logistic_regression.joblib"))
lr_metrics = evaluate(lr, X_test_s, y_test, "Logistic_Regression")

# -----------------------------
#  RANDOM FOREST
# -----------------------------
rf = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    n_jobs=-1,
    random_state=RANDOM_STATE
)
rf.fit(X_train, y_train)

joblib.dump(rf, os.path.join(OUT_DIR, "random_forest.joblib"))
rf_metrics = evaluate(rf, X_test, y_test, "Random_Forest")

# -----------------------------
# XGBOOST (OPTIONAL BUT STRONG)
# -----------------------------
xgb_metrics = None
if HAS_XGB:
    xgb_clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_STATE
    )
    xgb_clf.fit(X_train, y_train)

    joblib.dump(xgb_clf, os.path.join(OUT_DIR, "xgboost.joblib"))
    xgb_metrics = evaluate(xgb_clf, X_test, y_test, "XGBoost")

# -----------------------------
# SAVE SUMMARY
# -----------------------------
results = [lr_metrics, rf_metrics]
if xgb_metrics:
    results.append(xgb_metrics)

df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(OUT_DIR, "baseline_model_results.csv"), index=False)

print("\n BASELINE TRAINING COMPLETE")
print(df_results)
