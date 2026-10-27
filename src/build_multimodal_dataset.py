"""
Multi-Modal Dataset Builder for OASIS-3.
Fuses FreeSurfer structural MRI with UDS Health History (Vascular Burden)
and Family History (Genetics) to create an enriched, leak-free multimodal cohort.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

def build_multimodal_dataset(
    mri_csv="OASIS-3_Hippo_cdr_freesurfer_dataset_FIXED.csv",
    a5_csv="scans/UDSa5-Form_A5__Subject_Health_History/resources/csv/files/OASIS3_UDSa5_health_history.csv",
    a3_csv="scans/UDSa3-Form_A3__Subject_Family_History/resources/csv/files/OASIS3_UDSa3.csv",
    pup_csv="OASIS3_AV1451_PUP.csv",
    output_dir="data",
    test_size=0.20,
    random_state=42
):
    print("=" * 70)
    print("BUILDING MULTI-MODAL CLINICAL + NEUROIMAGING DATASET")
    print("=" * 70)

    # 1. Load Primary FreeSurfer MRI Data
    mri = pd.read_csv(mri_csv)
    print(f"Loaded MRI Dataset: {len(mri)} scans across {mri['Subject'].nunique()} subjects")

    # 2. Structural Feature Engineering
    mri["Normalized_Hippo"] = mri["TOTAL_HIPPOCAMPUS_VOLUME"] / mri["IntraCranialVol"]
    mri["Normalized_Cortex"] = mri["CortexVol"] / mri["IntraCranialVol"]
    mri["Normalized_SubCortGray"] = mri["SubCortGrayVol"] / mri["IntraCranialVol"]
    mri["Normalized_WhiteMatter"] = mri["CorticalWhiteMatterVol"] / mri["IntraCranialVol"]
    mri["Hippo_Asymmetry"] = (
        (mri["Left-Hippocampus_volume"] - mri["Right-Hippocampus_volume"]).abs()
        / mri["TOTAL_HIPPOCAMPUS_VOLUME"]
    )
    mri["Age"] = mri["age at visit"]

    # Normative hippocampal reference for z-scoring
    healthy_normals = mri[mri["cdr_total"] == 0]
    norm_hippo_mean = healthy_normals["Normalized_Hippo"].mean()
    norm_hippo_std = healthy_normals["Normalized_Hippo"].std()
    mri["Hippo_Z"] = (mri["Normalized_Hippo"] - norm_hippo_mean) / norm_hippo_std

    # Cognitive Reserve & Discordance Biomarker
    mri["MMSE_clean"] = mri["MMSE"].fillna(29.0)
    mri["Cognitive_Reserve_Buffer"] = mri["MMSE_clean"] / 30.0
    # Reserve Masked Index: Positive when hippocampus is atrophied but MMSE is artificially high
    mri["Reserve_Masked_Index"] = np.maximum(0.0, -mri["Hippo_Z"]) * mri["Cognitive_Reserve_Buffer"]

    # 3. Fuse Vascular Health History (Form A5)
    if os.path.exists(a5_csv):
        print("Fusing UDS Form A5 (Vascular Health History)...")
        a5 = pd.read_csv(a5_csv)
        a5["has_stroke"] = a5["CBSTROKE"].isin([1.0, 2.0]).astype(int)
        a5["has_hypertension"] = a5["HYPERTEN"].isin([1.0, 2.0]).astype(int)
        a5["has_heart_attack"] = a5["CVHATT"].isin([1.0, 2.0]).astype(int)

        sub_stroke = a5.groupby("OASISID")["has_stroke"].max().to_dict()
        sub_hyp = a5.groupby("OASISID")["has_hypertension"].max().to_dict()
        sub_cvhatt = a5.groupby("OASISID")["has_heart_attack"].max().to_dict()

        mri["Stroke_Risk"] = mri["Subject"].map(sub_stroke).fillna(0).astype(int)
        mri["Hypertension_Risk"] = mri["Subject"].map(sub_hyp).fillna(0).astype(int)
        mri["Heart_Disease_Risk"] = mri["Subject"].map(sub_cvhatt).fillna(0).astype(int)
        mri["Vascular_Burden_Index"] = (
            mri["Stroke_Risk"] + mri["Hypertension_Risk"] + mri["Heart_Disease_Risk"]
        )
    else:
        print("Warning: A5 not found, initializing vascular index to 0.")
        mri["Stroke_Risk"] = 0
        mri["Hypertension_Risk"] = 0
        mri["Heart_Disease_Risk"] = 0
        mri["Vascular_Burden_Index"] = 0

    # 4. Fuse Family Genetics History (Form A3)
    if os.path.exists(a3_csv):
        print("Fusing UDS Form A3 (Family Dementia History)...")
        a3 = pd.read_csv(a3_csv)
        mom_dem = a3.groupby("OASISID")["MOMDEM"].apply(lambda s: int((s == 1.0).any())).to_dict()
        dad_dem = a3.groupby("OASISID")["DADDEM"].apply(lambda s: int((s == 1.0).any())).to_dict()

        mri["Mom_Dementia"] = mri["Subject"].map(mom_dem).fillna(0).astype(int)
        mri["Dad_Dementia"] = mri["Subject"].map(dad_dem).fillna(0).astype(int)
        mri["Family_Dementia_History"] = (
            (mri["Mom_Dementia"] == 1) | (mri["Dad_Dementia"] == 1)
        ).astype(int)
    else:
        print("Warning: A3 not found, initializing family history to 0.")
        mri["Mom_Dementia"] = 0
        mri["Dad_Dementia"] = 0
        mri["Family_Dementia_History"] = 0

    # 5. Fuse Molecular Tau PET Pathology (Flortaucipir AV-1451 PUP)
    if os.path.exists(pup_csv):
        print("Fusing Flortaucipir AV-1451 Tau PET (PUP Regional SUVR)...")
        pup = pd.read_csv(pup_csv)
        pup["Subject"] = pup["PUP_PUPTIMECOURSEDATA ID"].str.extract(r"(OAS\d+)")
        pup["pup_days"] = pup["PUP_PUPTIMECOURSEDATA ID"].str.extract(r"_d(\d+)").astype(float)
        mri["mri_days"] = mri["days_from_entry"].astype(float)

        # Exact match on FreeSurfer Session ID (FSId)
        mri = pd.merge(
            mri,
            pup[["FSId", "PET_fSUVR_TOT_CTX_ENTORHINAL", "PET_fSUVR_TOT_CTX_INFERTMP", "PET_fSUVR_TOT_CORTMEAN"]],
            left_on="FS_FSDATA ID",
            right_on="FSId",
            how="left"
        )

        # Nearest visit match within 365 days for remaining subjects
        unmatched = mri[mri["PET_fSUVR_TOT_CTX_ENTORHINAL"].isna()].index
        if len(unmatched) > 0:
            mri_unmatched = mri.loc[unmatched].sort_values(["mri_days"]).reset_index()
            pup_sorted = pup.sort_values(["pup_days"]).reset_index(drop=True)
            m_asof = pd.merge_asof(
                mri_unmatched,
                pup_sorted[["Subject", "pup_days", "PET_fSUVR_TOT_CTX_ENTORHINAL", "PET_fSUVR_TOT_CTX_INFERTMP", "PET_fSUVR_TOT_CORTMEAN"]],
                left_on="mri_days",
                right_on="pup_days",
                by="Subject",
                direction="nearest",
                tolerance=365
            )
            for idx, orig_idx in enumerate(mri_unmatched["index"]):
                if pd.notna(m_asof.loc[idx, "PET_fSUVR_TOT_CTX_ENTORHINAL_y"]):
                    mri.loc[orig_idx, "PET_fSUVR_TOT_CTX_ENTORHINAL"] = m_asof.loc[idx, "PET_fSUVR_TOT_CTX_ENTORHINAL_y"]
                    mri.loc[orig_idx, "PET_fSUVR_TOT_CTX_INFERTMP"] = m_asof.loc[idx, "PET_fSUVR_TOT_CTX_INFERTMP_y"]
                    mri.loc[orig_idx, "PET_fSUVR_TOT_CORTMEAN"] = m_asof.loc[idx, "PET_fSUVR_TOT_CORTMEAN_y"]

        mri["Tau_PET_Available"] = mri["PET_fSUVR_TOT_CTX_ENTORHINAL"].notna().astype(int)
        # Normative baseline imputation for un-scanned visits (1.106 and 1.177)
        mri["Tau_Entorhinal_SUVR"] = mri["PET_fSUVR_TOT_CTX_ENTORHINAL"].fillna(1.106)
        mri["Tau_InferiorTemporal_SUVR"] = mri["PET_fSUVR_TOT_CTX_INFERTMP"].fillna(1.177)
    else:
        print("Warning: AV1451 PUP not found, initializing tau features with baseline reference.")
        mri["Tau_PET_Available"] = 0
        mri["Tau_Entorhinal_SUVR"] = 1.106
        mri["Tau_InferiorTemporal_SUVR"] = 1.177

    # 6. Target Definition: Alzheimer's Disease Spectrum (CDR >= 0.5)
    mri["target"] = (mri["cdr_total"] >= 0.5).astype(int)

    # Core Multi-Modal Features
    multimodal_features = [
        "Age",
        "Normalized_Hippo",
        "Normalized_Cortex",
        "Normalized_SubCortGray",
        "Normalized_WhiteMatter",
        "Hippo_Asymmetry",
        "Reserve_Masked_Index",
        "Cognitive_Reserve_Buffer",
        "Vascular_Burden_Index",
        "Family_Dementia_History",
        "Tau_Entorhinal_SUVR",
        "Tau_InferiorTemporal_SUVR",
        "Tau_PET_Available"
    ]

    metadata_cols = [
        "Subject",
        "Session",
        "days_from_entry",
        "cdr_total",
        "CDRSUM",
        "conversion_status",
        "dx1",
        "MMSE",
        "TOTAL_HIPPOCAMPUS_VOLUME",
        "IntraCranialVol",
        "Stroke_Risk",
        "Hypertension_Risk"
    ]

    clean_df = mri[metadata_cols + ["target"] + multimodal_features].copy()

    # 7. Patient-Level Train/Holdout Partition (Strict Zero Leakage)
    os.makedirs(output_dir, exist_ok=True)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(clean_df, clean_df["target"], groups=clean_df["Subject"]))

    train_df = clean_df.iloc[train_idx].copy().reset_index(drop=True)
    test_df = clean_df.iloc[test_idx].copy().reset_index(drop=True)

    # Verify zero subject leakage
    overlap = set(train_df["Subject"]).intersection(set(test_df["Subject"]))
    assert len(overlap) == 0, f"Error: Subject leakage detected: {overlap}"

    print(f"\n--- Multi-Modal Partition Summary ---")
    print(f"Train Set:   {len(train_df)} scans across {train_df['Subject'].nunique()} subjects")
    print(f"Holdout Set: {len(test_df)} scans across {test_df['Subject'].nunique()} subjects")
    print(f"Vascular Burden > 0 in Holdout: {(test_df['Vascular_Burden_Index'] > 0).sum()} scans")
    print(f"Family History > 0 in Holdout:  {(test_df['Family_Dementia_History'] > 0).sum()} scans")
    print(f"Tau PET Available in Holdout:    {(test_df['Tau_PET_Available'] > 0).sum()} scans")

    train_path = os.path.join(output_dir, "oasis3_multimodal_train.csv")
    test_path = os.path.join(output_dir, "oasis3_multimodal_test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Saved: {train_path}")
    print(f"Saved: {test_path}")

    return train_path, test_path

if __name__ == "__main__":
    build_multimodal_dataset()
