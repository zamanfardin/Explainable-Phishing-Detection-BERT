import pandas as pd
import re
import os

# -----------------------------
# PATHS
# -----------------------------
RAW_DATA_PATH = "Users/fardinzaman/Desktop/New_Dataset/raw_email_dataset.csv"   # Path to your raw CSV file
OUTPUT_PATH = "Users/fardinzaman/Desktop/New_Dataset/Research_Dataset_Cleaned.csv"  # Path to save cleaned CSV

# -----------------------------
# LOAD DATA
# -----------------------------
if not os.path.exists(RAW_DATA_PATH):
    raise FileNotFoundError(f"File not found: {RAW_DATA_PATH}")

df = pd.read_csv(RAW_DATA_PATH)

# -----------------------------
# STANDARDISE COLUMN NAMES
# -----------------------------
df.columns = [c.lower().strip() for c in df.columns]

# Expecting: 'text' and 'label'
assert "text" in df.columns, "Missing 'text' column"
assert "label" in df.columns, "Missing 'label' column"

# -----------------------------
# DROP NULLS & DUPLICATES
# -----------------------------
df = df.dropna(subset=["text", "label"])
df = df.drop_duplicates(subset=["text"])

# -----------------------------
# LABEL NORMALISATION
# -----------------------------
def normalize_label(label):
    """
    Convert various label formats to 0 (not phishing) or 1 (phishing)
    """
    label = str(label).lower()
    if label in ["phishing", "spam", "1"]:
        return 1
    else:
        return 0

df["label"] = df["label"].apply(normalize_label)

# -----------------------------
# TEXT CLEANING FUNCTION
# -----------------------------
def clean_text(text):
    """
    Lowercase, remove URLs, emails, HTML, non-alphanumeric chars, extra spaces
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)   # remove URLs
    text = re.sub(r"\S+@\S+", "", text)          # remove emails
    text = re.sub(r"<.*?>", "", text)            # remove HTML tags
    text = re.sub(r"[^a-z0-9\s!?]", " ", text)  # remove symbols except !?
    text = re.sub(r"\s+", " ", text).strip()    # normalize spaces
    return text

df["text"] = df["text"].apply(clean_text)

# -----------------------------
# FINAL SANITY CHECK
# -----------------------------
print(" Final dataset shape:", df.shape)
print("Class distribution:")
print(df["label"].value_counts())

# -----------------------------
# SAVE CLEAN DATASET
# -----------------------------
df.to_csv(OUTPUT_PATH, index=False)
print(f" Cleaned dataset saved to: {OUTPUT_PATH}")
