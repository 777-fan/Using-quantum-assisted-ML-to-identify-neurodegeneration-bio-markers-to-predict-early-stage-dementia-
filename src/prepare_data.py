"""
Data preparation and subject-level train/holdout split for OASIS-3.
Ensures zero data leakage across longitudinal visits for the same subject.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import os

def prepare_oasis3_data(
    input_csv="OASIS-3_Hippo_cdr_freesurfer_dataset_FIXED.csv",
    output_dir="data",
    test_size=0.20,
    random_state=42
):
    print(f"Loading raw dataset from {input_csv}...")
    df = pd.read_csv(input_csv)
    print(f"Total rows: {len(df)}, Unique subjects: {df['Subject'].nunique()}")

    # 1. Feature Engineering: Intracranial Volume (ICV) normalization
    df["Normalized_Hippo"] = df["TOTAL_HIPPOCAMPUS_VOLUME"] / df["IntraCranialVol"]
    df["Normalized_Cortex"] = df["CortexVol"] / df["IntraCranialVol"]
    df["Normalized_SubCortGray"] = df["SubCortGrayVol"] / df["IntraCranialVol"]
    df["Normalized_WhiteMatter"] = df["CorticalWhiteMatterVol"] / df["IntraCranialVol"]
    df["Hippo_Asymmetry"] = (
        (df["Left-Hippocampus_volume"] - df["Right-Hippocampus_volume"]).abs() 
        / df["TOTAL_HIPPOCAMPUS_VOLUME"]
    )
    df["Age"] = df["age at visit"]

    # 2. Target Definition: Alzheimer's Spectrum Impairment (CDR >= 0.5)
    # 0 = Cognitively Normal (CDR 0.0)
    # 1 = Impaired / AD Spectrum (CDR 0.5, 1.0, 2.0)
    df["target"] = (df["cdr_total"] >= 0.5).astype(int)

    # Key feature columns used by the models
    feature_cols = [
        "Age",
        "Normalized_Hippo",
        "Normalized_Cortex",
        "Normalized_SubCortGray",
        "Normalized_WhiteMatter",
        "Hippo_Asymmetry",
        "TOTAL_HIPPOCAMPUS_VOLUME",
        "IntraCranialVol"
    ]

    # Clinical/Tracking metadata columns
    meta_cols = [
        "Subject",
        "Session",
        "days_from_entry",
        "cdr_total",
        "CDRSUM",
        "conversion_status",
        "dx1",
        "MMSE"
    ]

    all_cols = meta_cols + ["target"] + feature_cols
    df_clean = df[all_cols].copy()

    # 3. Patient-Level Train/Holdout Split (Grouped by Subject)
    os.makedirs(output_dir, exist_ok=True)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(df_clean, df_clean["target"], groups=df_clean["Subject"]))

    train_df = df_clean.iloc[train_idx].copy().reset_index(drop=True)
    test_df = df_clean.iloc[test_idx].copy().reset_index(drop=True)

    # Verify 0% patient leakage
    train_subjects = set(train_df["Subject"].unique())
    test_subjects = set(test_df["Subject"].unique())
    overlap = train_subjects.intersection(test_subjects)
    assert len(overlap) == 0, f"Error: Subject leakage detected! Overlapping subjects: {overlap}"

    print(f"\n--- Split Summary ---")
    print(f"Train Set: {len(train_df)} scans across {len(train_subjects)} unique subjects")
    print(f"  Class balance: Normal (0) = {(train_df['target'] == 0).sum()}, Impaired (1) = {(train_df['target'] == 1).sum()}")
    print(f"Holdout Test Set: {len(test_df)} scans across {len(test_subjects)} unique subjects")
    print(f"  Class balance: Normal (0) = {(test_df['target'] == 0).sum()}, Impaired (1) = {(test_df['target'] == 1).sum()}")
    print(f"Subject Leakage Check: PASSED (0 overlapping subjects)")

    train_path = os.path.join(output_dir, "oasis3_train_patients.csv")
    test_path = os.path.join(output_dir, "oasis3_holdout_test_patients.csv")
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Saved: {train_path}")
    print(f"Saved: {test_path}")

    return train_path, test_path

if __name__ == "__main__":
    prepare_oasis3_data()
