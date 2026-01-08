# train_combined_embeddings.py
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Optional: XGBoost
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ------------------------
# CONFIG
# ------------------------
OUT_DIR = "/Users/fardinzaman/Desktop/New_Dataset/phish_results_combined"
os.makedirs(OUT_DIR, exist_ok=True)
RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15  # relative to trainval

# Paths to embeddings
BERT_EMB_PATH = "/Users/fardinzaman/Desktop/New_Dataset/Email_Embeddings.npy"
DISTIL_EMB_PATH = "/Users/fardinzaman/Desktop/New_Dataset/DistilBERT_Email_Embeddings.npy"
LABEL_PATH = "/Users/fardinzaman/Desktop/New_Dataset/Distil_Email_Labels.npy"

# ------------------------
# HELPER FUNCTIONS
# ------------------------
def evaluate_model(model, Xs, ys, name):
    y_pred = model.predict(Xs)
    y_prob = model.predict_proba(Xs)[:,1] if hasattr(model, "predict_proba") else np.zeros(len(y_pred))
    acc = accuracy_score(ys, y_pred)
    prec = precision_score(ys, y_pred)
    rec = recall_score(ys, y_pred)
    f1 = f1_score(ys, y_pred)
    roc = roc_auc_score(ys, y_prob)
    
    # Save classification report
    report = classification_report(ys, y_pred, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(os.path.join(OUT_DIR, f"{name}_classification_report.csv"))
    
    # Confusion matrix plot
    cm = confusion_matrix(ys, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(os.path.join(OUT_DIR, f"{name}_confusion_matrix.png"), bbox_inches='tight')
    plt.close()
    
    return {"model": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": roc}

def train_and_evaluate(X, y, model_name_prefix):
    # Split train/val/test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    relative_val = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=relative_val, stratify=y_trainval, random_state=RANDOM_STATE
    )

    print(f"\nShapes -> train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")

    # Scale features for Logistic Regression
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(OUT_DIR, f"{model_name_prefix}_scaler.joblib"))

    results = []

    # ---------- Logistic Regression ----------
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=RANDOM_STATE)
    lr.fit(X_train_s, y_train)
    joblib.dump(lr, os.path.join(OUT_DIR, f"{model_name_prefix}_logistic.joblib"))
    results.append(evaluate_model(lr, X_val_s, y_val, f"{model_name_prefix}_Logistic_val"))
    results.append(evaluate_model(lr, X_test_s, y_test, f"{model_name_prefix}_Logistic_test"))

    # ---------- Random Forest ----------
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', n_jobs=-1, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    joblib.dump(rf, os.path.join(OUT_DIR, f"{model_name_prefix}_rf.joblib"))
    results.append(evaluate_model(rf, X_val, y_val, f"{model_name_prefix}_RF_val"))
    results.append(evaluate_model(rf, X_test, y_test, f"{model_name_prefix}_RF_test"))

    # ---------- XGBoost ----------
    xgb_clf = None
    if HAS_XGB:
        print("\nTraining XGBoost...")
        xgb_clf = xgb.XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE)
        xgb_clf.fit(X_train, y_train)
        joblib.dump(xgb_clf, os.path.join(OUT_DIR, f"{model_name_prefix}_xgb.joblib"))
        results.append(evaluate_model(xgb_clf, X_val, y_val, f"{model_name_prefix}_XGB_val"))
        results.append(evaluate_model(xgb_clf, X_test, y_test, f"{model_name_prefix}_XGB_test"))

    # Save metrics summary
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUT_DIR, f"{model_name_prefix}_metrics_summary.csv"), index=False)

    # ---------- SHAP Explainability ----------
    best_tree = rf if rf is not None else (xgb_clf if HAS_XGB else None)
    if best_tree:
        print(f"\nRunning SHAP explainability on {best_tree.__class__.__name__}...")
        shap_sample_size = min(500, X_test.shape[0])
        idxs = np.random.RandomState(RANDOM_STATE).choice(np.arange(X_test.shape[0]), shap_sample_size, replace=False)
        X_shap = X_test[idxs]

        explainer = shap.TreeExplainer(best_tree, feature_perturbation='tree_path_dependent')
        shap_values = explainer.shap_values(X_shap, check_additivity=False)
        shap_to_plot = shap_values[1] if isinstance(shap_values, list) else shap_values

        plt.figure(figsize=(10,6))
        shap.summary_plot(shap_to_plot, X_shap, show=False, plot_type="bar")
        plt.savefig(os.path.join(OUT_DIR, f"{model_name_prefix}_shap_summary_bar.png"), bbox_inches='tight')
        plt.close()

    return results_df

# ------------------------
# RUN TRAINING
# ------------------------
print("\n===== Loading embeddings =====")
bert_emb = np.load(BERT_EMB_PATH)
distil_emb = np.load(DISTIL_EMB_PATH)
y = np.load(LABEL_PATH)

print("===== Combining BERT + DistilBERT embeddings =====")
X_combined = np.concatenate([bert_emb, distil_emb], axis=1)  # shape -> (n_samples, 768*2)

print("===== Training on combined embeddings =====")
train_and_evaluate(X_combined, y, "BERT+DistilBERT")

print("\n Combined embeddings training and explainability complete!")
