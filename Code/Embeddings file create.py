import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
from tqdm import tqdm  

# -----------------------------
# CONFIGURATION
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_CHUNK_LEN = 512
BATCH_SIZE = 1  # Emails have variable chunk lengths

# -----------------------------
# LOAD PREPROCESSED DATA
# -----------------------------
pkl_path = "/Users/fardinzaman/Desktop/New_Dataset/Research_Processed_Hierarchical.pkl"
df = pd.read_pickle(pkl_path)
print(f"Loaded {len(df)} emails")


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.to(DEVICE)
bert_model.eval()  

# -----------------------------
# CUSTOM DATASET
# -----------------------------
class EmailDataset(Dataset):
    def __init__(self, df):
        self.emails = df["chunks"].tolist()  # list of chunk lists
        self.labels = df["label"].tolist()

    def __len__(self):
        return len(self.emails)

    def __getitem__(self, idx):
        return self.emails[idx], self.labels[idx]

dataset = EmailDataset(df)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------------
# FUNCTION TO GET CLS EMBEDDINGS
# -----------------------------
def get_cls_embeddings(email_chunks):
    """
    Converts a list of chunks into a single email-level embedding.
    Handles lists and torch tensors, skips invalid/empty chunks.
    """
    chunk_embeddings = []

    with torch.no_grad():
        for chunk in email_chunks:
            # Handle both list and tensor
            if isinstance(chunk, torch.Tensor):
                chunk_ids = chunk.tolist()
            elif isinstance(chunk, list):
                chunk_ids = chunk
            else:
                print(f"⚠️ Skipped one email due to error: Unexpected chunk type: {type(chunk)}")
                continue

            # Skip empty chunks
            if len(chunk_ids) == 0:
                continue

            # Convert to tensor and pad to MAX_CHUNK_LEN
            input_ids = torch.tensor([chunk_ids], dtype=torch.long).to(DEVICE)
            seq_len = input_ids.shape[1]
            if seq_len < MAX_CHUNK_LEN:
                pad_len = MAX_CHUNK_LEN - seq_len
                pad_tensor = torch.full((1, pad_len), tokenizer.pad_token_id, dtype=torch.long).to(DEVICE)
                input_ids = torch.cat([input_ids, pad_tensor], dim=1)

            attention_mask = (input_ids != tokenizer.pad_token_id).long().to(DEVICE)

            # BERT forward pass
            outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
            chunk_embeddings.append(cls_embedding.cpu().numpy())

    if len(chunk_embeddings) == 0:
        # Return zero vector if all chunks invalid
        return np.zeros(bert_model.config.hidden_size)

    # Mean pooling across chunks
    email_embedding = np.mean(np.vstack(chunk_embeddings), axis=0)
    return email_embedding

# -----------------------------
# EXTRACT EMAIL-LEVEL EMBEDDINGS
# -----------------------------
email_embeddings = []
labels = []

print("Extracting email-level embeddings (this may take some time)...")
for chunks, label in tqdm(dataloader, total=len(dataloader)):
    emb = get_cls_embeddings(chunks[0])  # batch_size=1
    email_embeddings.append(emb)
    labels.append(label.item())

email_embeddings = np.vstack(email_embeddings)
labels = np.array(labels)
print(f"\n Email-level embeddings ready: {email_embeddings.shape}")

# -----------------------------
# SAVE EMBEDDINGS AND LABELS
# -----------------------------
np.save("/Users/fardinzaman/Desktop/New_Dataset/Email_Embeddings.npy", email_embeddings)
np.save("/Users/fardinzaman/Desktop/New_Dataset/Email_Labels.npy", labels)
print("Saved embeddings and labels successfully")

# -----------------------------
# SUMMARY REPORT
# -----------------------------
unique, counts = np.unique(labels, return_counts=True)
print("\nClass distribution:")
for u, c in zip(unique, counts):
    print(f"Label {u}: {c} ({c/len(labels)*100:.2f}%)")

zero_vectors = np.sum(np.all(email_embeddings == 0, axis=1))
print(f"Number of emails with zero embeddings: {zero_vectors}")
