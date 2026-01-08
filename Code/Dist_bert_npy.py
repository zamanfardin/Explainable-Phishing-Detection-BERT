import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizerFast, BertModel
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
PKL_PATH = "/Users/fardinzaman/Desktop/New_Dataset/Research_Processed_Hierarchical.pkl" 
OUT_EMB = "/Users/fardinzaman/Desktop/New_Dataset/Distil_Email_Labels.npy"
OUT_LAB = "/Users/fardinzaman/Desktop/New_Dataset/DistilBERT_Email_Embeddings.npy"

CHUNK_SIZE = 512  # BERT max tokens
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_pickle(PKL_PATH)
print(f"Loaded {len(df)} emails")

TEXT_COLUMN = "text"  
LABEL_COLUMN = "label" 

texts = df[TEXT_COLUMN].astype(str).tolist()
labels = df[LABEL_COLUMN].values

# -----------------------------
# LOAD BERT
# -----------------------------
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.to(DEVICE)
model.eval()

# -----------------------------
# HELPER FUNCTION
# -----------------------------
def chunk_text(text, chunk_size=CHUNK_SIZE):
    tokens = tokenizer.encode(text, add_special_tokens=True)
    chunks = [tokens[i:i+chunk_size] for i in range(0, len(tokens), chunk_size)]
    return chunks

def encode_chunks(chunks):
    embeddings = []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i+BATCH_SIZE]
        batch_padded = tokenizer.pad(
            {"input_ids": batch},
            padding=True,
            return_tensors="pt"
        ).to(DEVICE)
        with torch.no_grad():
            output = model(**batch_padded)
            # Take CLS token embedding
            batch_emb = output.last_hidden_state[:, 0, :]
            embeddings.append(batch_emb.cpu().numpy())
    return np.vstack(embeddings)

# -----------------------------
# GENERATE HIERARCHICAL EMBEDDINGS
# -----------------------------
all_embeddings = []

print("Generating Hierarchical BERT embeddings...")
for text in tqdm(texts):
    chunks = chunk_text(text)
    email_emb = encode_chunks(chunks).mean(axis=0)  # mean of chunk embeddings
    all_embeddings.append(email_emb)

X = np.vstack(all_embeddings)
y = np.array(labels)

# -----------------------------
# SAVE OUTPUTS
# -----------------------------
np.save(OUT_EMB, X)
np.save(OUT_LAB, y)

print(" HierBERT embedding extraction complete")
print("Embeddings shape:", X.shape)
print("Labels shape:", y.shape)
