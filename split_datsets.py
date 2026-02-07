import os
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = "/home/lucifer/projects/Job-Recommender/datasets"
SPLIT_DIR = os.path.join(BASE_DIR, "Splits")

FILES = [
    "applications.csv",
    "candidates.csv",
    "jobs.csv"
]

# Create split directories
for split in ["train", "cv", "test"]:
    os.makedirs(os.path.join(SPLIT_DIR, split), exist_ok=True)

def split_and_save(csv_file):
    path = os.path.join(BASE_DIR, csv_file)
    df = pd.read_csv(path)

    # 70% train, 30% temp
    train_df, temp_df = train_test_split(
        df,
        test_size=0.3,
        random_state=42,
        shuffle=True
    )

    # split temp into 20% cv, 10% test  -> (2/3, 1/3)
    cv_df, test_df = train_test_split(
        temp_df,
        test_size=1/3,
        random_state=42,
        shuffle=True
    )

    train_df.to_csv(os.path.join(SPLIT_DIR, "train", csv_file), index=False)
    cv_df.to_csv(os.path.join(SPLIT_DIR, "cv", csv_file), index=False)
    test_df.to_csv(os.path.join(SPLIT_DIR, "test", csv_file), index=False)

    print(f"✅ Split completed for {csv_file}")
    print(f"   Train: {len(train_df)}, CV: {len(cv_df)}, Test: {len(test_df)}")

for file in FILES:
    split_and_save(file)
