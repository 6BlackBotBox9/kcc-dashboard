import pandas as pd
import os

# 1. Dataset Path
data_path = r"C:\Users\Priyaranjan\Downloads\kcc_dashboard\kcc_dashboard\datasetsQueries\Question-Answer\questionsv4kaggle.csv"

if not os.path.exists(data_path):
    print(f"Error: Could not find file at {data_path}")
    exit(1)

print("Loading dataset for analysis...\n")
df = pd.read_csv(data_path)

# Ensure valid text in questions
df_valid = df.dropna(subset=['questions']).copy()

# 2. Count Duplicates
raw_duplicates = df_valid.duplicated(subset=['questions']).sum()

normalized_queries = df_valid['questions'].astype(str).str.strip().str.lower()
norm_duplicates = normalized_queries.duplicated().sum()

# Check conflicting answers (if 'answers' column exists)
has_answers = 'answers' in df_valid.columns
if has_answers:
    df_valid['norm_q'] = normalized_queries
    unique_answers_per_q = df_valid.groupby('norm_q')['answers'].nunique()
    conflicting_queries_count = (unique_answers_per_q > 1).sum()

# 3. Print Terminal Report
print("=" * 60)
print("             DATASET DUPLICATE ANALYSIS REPORT               ")
print("=" * 60)
print(f"Total Rows Evaluated            : {len(df_valid)}")
print(f"Exact Text Duplicates           : {raw_duplicates}")
print(f"Normalized Duplicates (Case/Space): {norm_duplicates}")
if has_answers:
    print(f"Queries with Conflicting Answers: {conflicting_queries_count}")
print("=" * 60 + "\n")

print("Top 5 Most Repeated Queries in Dataset:\n")
top_repeats = df_valid['questions'].value_counts().head(5)
for query, count in top_repeats.items():
    print(f" - [{count} times]: \"{query}\"")