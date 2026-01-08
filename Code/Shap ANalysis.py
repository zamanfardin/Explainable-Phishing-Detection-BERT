

import numpy as np
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import json
import os

# -------------------------
# PATHS
# -------------------------
MODEL_PATH = "Users/fardinzaman/Desktop/New_Dataset/4.4phish_results/BERT_logistic.joblib"           # change if RF
EMB_PATH = "Users/fardinzaman/Desktop/New_Dataset/Combined_Embeddings.npy"          # BERT / Distil / Combined
LABEL_PATH = "Users/fardinzaman/Desktop/New_Dataset/Email_Labels.npy"
LINGUISTIC_PATH = "Users/fardinzaman/Desktop/New_Dataset/linguistic_features.csv"
OUTPUT_DIR = "Users/fardinzaman/Desktop/New_Dataset/SHAP"

os.makedirs(OUTPUT_DIR, exist_ok=True)
shap.initjs()

# -------------------------
# LOAD DATA
# -------------------------
print("Loading model and data...")

model = joblib.load(MODEL_PATH)
X = np.load(EMB_PATH)
y = np.load(LABEL_PATH)

feature_names = [f"emb_{i}" for i in range(X.shape[1])]

# Sample for SHAP efficiency
X_sample = X[:1000]

# -------------------------
# SHAP EXPLAINER
# -------------------------
explainer = shap.Explainer(model, X_sample, feature_names=feature_names)
shap_values = explainer(X_sample)

# -------------------------
# 1️⃣ SHAP SUMMARY PLOT
# -------------------------
plt.figure()
shap.summary_plot(shap_values, X_sample, show=False)
plt.savefig(f"{OUTPUT_DIR}/Figure_SHAP_1_Summary_Plot.png", bbox_inches="tight")
plt.close()

# -------------------------
# 2️⃣ FEATURE IMPORTANCE
# -------------------------
plt.figure()
shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
plt.savefig(f"{OUTPUT_DIR}/Figure_SHAP_2_Feature_Importance.png", bbox_inches="tight")
plt.close()

# -------------------------
# 3️⃣ WATERFALL – PHISHING
# -------------------------
phish_idx = np.where(y[:1000] == 1)[0][0]

plt.figure()
shap.plots.waterfall(shap_values[phish_idx], max_display=15, show=False)
plt.savefig(f"{OUTPUT_DIR}/Figure_SHAP_3_Waterfall_Spam.png", bbox_inches="tight")
plt.close()

# -------------------------
# 4️⃣ WATERFALL – HAM
# -------------------------
ham_idx = np.where(y[:1000] == 0)[0][0]

plt.figure()
shap.plots.waterfall(shap_values[ham_idx], max_display=15, show=False)
plt.savefig(f"{OUTPUT_DIR}/Figure_SHAP_4_Waterfall_Ham.png", bbox_inches="tight")
plt.close()

# -------------------------
# 5️⃣ LINGUISTIC FEATURE SHAP
# -------------------------
if os.path.exists(LINGUISTIC_PATH):
    print("Processing linguistic features...")
    
    ling_df = pd.read_csv(LINGUISTIC_PATH)
    ling_features = ling_df.values[:1000]
    ling_names = ling_df.columns.tolist()

    ling_explainer = shap.Explainer(model, ling_features)
    ling_shap = ling_explainer(ling_features)

    plt.figure()
    shap.summary_plot(ling_shap, ling_features, feature_names=ling_names, show=False)
    plt.savefig(f"{OUTPUT_DIR}/Figure_SHAP_5_Linguistic_Summary.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(ling_shap, ling_features, feature_names=ling_names,
                      plot_type="bar", show=False)
    plt.savefig(f"{OUTPUT_DIR}/Figure_SHAP_6_Linguistic_Importance.png", bbox_inches="tight")
    plt.close()

# -------------------------
# 6️⃣ SAVE SHAP VALUES
# -------------------------
np.save(f"{OUTPUT_DIR}/shap_values_rf.npy", shap_values.values)

# -------------------------
# 7️⃣ TOKEN-LEVEL SHAP (SIMULATED)
# -------------------------
token_analysis = {
    "sample_id": int(phish_idx),
    "tokens": feature_names[:20],
    "shap_values": shap_values.values[phish_idx][:20].tolist()
}

with open(f"{OUTPUT_DIR}/token_level_analysis.json", "w") as f:
    json.dump(token_analysis, f, indent=4)

print("SHAP & Unified Interpretation completed successfully.")
