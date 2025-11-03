import pandas as pd
DATA_PATH = "/Users/fardinzaman/Desktop/Dataset/processed_data.csv"
df = pd.read_csv(DATA_PATH)

print(f"Original dataset shape: {df.shape}")
print("Columns:", df.columns)


df['subject'] = df['subject'].fillna('')

# Combine subject and body into a new 'text' 
df['text'] = df['subject'] + ' ' + df['body']
print("Null values in combined text:", df['text'].isnull().sum())

df = df.dropna(subset=['body'])
print(f"Dataset shape after combining subject + body: {df.shape}")

df.to_csv("/Users/fardinzaman/Desktop/Dataset/Research_Dataset_Final.csv", index=False)
print("Combined dataset saved to 'Research_Dataset_Final.csv'")
