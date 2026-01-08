import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

# Optional XGBoost
try:
    import xgboost as xgb
    HAS_XGB = True
except:
    HAS_XGB = False

# -----------------------------
# PATHS
# -----------------------------
EMB_PATH = "C:/Users/u1177867/Desktop/Email_Embeddings.npy"
LBL_PATH = "C:/Users/u1177867/Desktop/Email_Labels.npy"

# -----------------------------
# LOAD DATA
# -----------------------------
X = np.load(EMB_PATH)
y = np.load(LBL_PATH)

print("Loaded embeddings:", X.shape)
print("Loaded labels:", y.shape)

# -----------------------------
# TRAIN / TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# -----------------------------
# SCALE FOR LINEAR MODELS
# -----------------------------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# -----------------------------
# HELPER FUNCTION
# -----------------------------
def evaluate(model, X, y, name):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    return {
        "Model": name,
        "Accuracy": accuracy_score(y, y_pred),
        "Precision": precision_score(y, y_pred),
        "Recall": recall_score(y, y_pred),
        "F1": f1_score(y, y_pred),
        "ROC_AUC": roc_auc_score(y, y_prob)
    }

results = []

# -----------------------------
# LOGISTIC REGRESSION (Baseline)
# -----------------------------
lr = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)
lr.fit(X_train_s, y_train)

results.append(evaluate(lr, X_test_s, y_test, "Logistic Regression"))

# -----------------------------
# RANDOM FOREST
# -----------------------------
rf = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)

results.append(evaluate(rf, X_test, y_test, "Random Forest"))

# -----------------------------
#  XGBOOST (OPTIONAL)
# -----------------------------
if HAS_XGB:
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
    xgb_model.fit(X_train, y_train)

    results.append(evaluate(xgb_model, X_test, y_test, "XGBoost"))

# -----------------------------
# RESULTS TABLE
# -----------------------------
results_df = pd.DataFrame(results)
print("\n BASELINE RESULTS")
print(results_df)

results_df.to_csv("baseline_results.csv", index=False)
print("\n Saved baseline_results.csv")
