import pandas as pd
import os

# 1. Dataset Paths
data_path = r"C:\Users\Priyaranjan\Downloads\kcc_dashboard\kcc_dashboard\datasetsQueries\Question-Answer\questionsv4kaggle.csv"
output_path = r"C:\Users\Priyaranjan\Downloads\kcc_dashboard\kcc_dashboard\datasetsQueries\Question-Answer\questionsv4kaggle_cleaned.csv"

if not os.path.exists(data_path):
    print(f"Error: Could not find file at {data_path}")
    exit(1)

print("Loading dataset for cleaning...")
df = pd.read_csv(data_path).dropna(subset=['questions', 'answers'])
original_count = len(df)

# 2. Normalize text to catch case and spacing variations
df['norm_q'] = df['questions'].astype(str).str.strip().str.lower()
df['norm_a'] = df['answers'].astype(str).str.strip().str.lower()

# 3. Deduplicate: Drop ONLY when both question AND answer match
df_cleaned = df.drop_duplicates(subset=['norm_q', 'norm_a'], keep='first').copy()

# 4. Remove temporary normalization columns
df_cleaned = df_cleaned.drop(columns=['norm_q', 'norm_a'])
cleaned_count = len(df_cleaned)

# 5. Save the cleaned dataset
df_cleaned.to_csv(output_path, index=False)

print("\n" + "="*60)
print("             DATASET CLEANING SUMMARY REPORT               ")
print("="*60)
print(f"Original Row Count              : {original_count}")
print(f"Cleaned Row Count               : {cleaned_count}")
print(f"Redundant Rows Removed          : {original_count - cleaned_count}")
print(f"Saved Output File               : questionsv4kaggle_cleaned.csv")
print("="*60 + "\n")