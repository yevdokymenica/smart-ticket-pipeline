
from pathlib import Path
import pandas as pd

# Define dataset location
DATA_PATH = Path(__file__).resolve().parent / "data" / "tickets_dataset.csv"

# Check whether the file exists
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

# Load dataset
df = pd.read_csv(DATA_PATH)

# Check required columns
required_columns = {"ticket_text", "category"}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

# Dataset overview
print("\n--- DATASET OVERVIEW ---")
print(f"Number of rows: {len(df)}")
print(f"Number of columns: {len(df.columns)}")
print(f"Columns: {list(df.columns)}")

# Check missing values
print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

# Check duplicate tickets
duplicates = df.duplicated(subset=["ticket_text"]).sum()
print(f"\nDuplicate tickets: {duplicates}")

# Check category distribution
print("\n--- CATEGORY DISTRIBUTION ---")
print(df["category"].value_counts())

# Check empty texts
empty_texts = df["ticket_text"].fillna("").str.strip().eq("").sum()
print(f"\nEmpty ticket texts: {empty_texts}")

print("\n[SUCCESS] Dataset inspection completed!")
