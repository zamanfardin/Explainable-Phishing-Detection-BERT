
import numpy as np
import os

# ------------------------
# PATHS
# ------------------------
BERT_EMB_PATH = "/Users/fardinzaman/Desktop/New_Dataset/Email_Embeddings.npy"
DISTIL_EMB_PATH = "/Users/fardinzaman/Desktop/New_Dataset/DistilBERT_Email_Embeddings.npy"
OUT_PATH = "/Users/fardinzaman/Desktop/New_Dataset/Combined_Embeddings.npy"

print("Loading BERT embeddings...")
bert_emb = np.load(BERT_EMB_PATH)

print("Loading DistilBERT embeddings...")
distil_emb = np.load(DISTIL_EMB_PATH)

# ------------------------
# SAFETY CHECKS
# ------------------------
assert bert_emb.shape[0] == distil_emb.shape[0], "Sample size mismatch!"
assert bert_emb.shape[1] == 768, "BERT embedding size incorrect!"
assert distil_emb.shape[1] == 768, "DistilBERT embedding size incorrect!"

print(f"BERT shape: {bert_emb.shape}")
print(f"DistilBERT shape: {distil_emb.shape}")

# ------------------------
# COMBINE (CONCATENATE)
# ------------------------
combined_emb = np.concatenate([bert_emb, distil_emb], axis=1)

print(f"Combined embeddings shape: {combined_emb.shape}")

# ------------------------
# SAVE
# ------------------------
np.save(OUT_PATH, combined_emb)

print(f"✅ Combined embeddings saved to:\n{OUT_PATH}")
