"""
Cohort-Wide Neurodegeneration Biomarker Discovery Engine.
Runs the pre-trained multi-modal quantum model across all 442 unseen holdout patients,
calculates exact TreeSHAP attribution values for every patient, and exports:
1. reports/cohort_biomarker_discovery.csv (Full spreadsheet of predictions & primary biomarkers)
2. reports/global_biomarker_importance.png (Publication-grade biomarker ranking plot)
"""

import os
import sys
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import xgboost as xgb
from src.unified_quantum_model import UnifiedMultiModalQuantumModel

def main():
    print("=" * 80)
    print("COHORT-WIDE NEURODEGENERATION BIOMARKER DISCOVERY ENGINE")
    print("=" * 80)

    test_csv = "data/oasis3_multimodal_test.csv"
    if not os.path.exists(test_csv):
        print(f"Error: {test_csv} not found.")
        return

    test_df = pd.read_csv(test_csv)
    print(f"Loaded {len(test_df)} unseen holdout patient scans ({test_df['Subject'].nunique()} unique subjects).")

    # 1. Load Pre-Trained Multi-Modal Model
    print("Loading pre-trained Multi-Modal Quantum-Embedded model...")
    model = UnifiedMultiModalQuantumModel.load("saved_models")

    # 2. Extract Features & Transform to Unified 16D Space
    X_raw = test_df[model.classical_features].values
    print("Extracting 6-Qubit PennyLane Quantum Entanglement embeddings...")
    X_unified = model.transform_to_unified(X_raw)

    # 3. Predict Multi-Modal Dementia Risk for All Patients
    probs = model.xgb_model.predict_proba(X_unified)[:, 1]

    # 4. Compute Exact TreeSHAP Attribution for All 442 Patients (C++ compiled)
    print("Computing TreeSHAP local attribution across all patients...")
    booster = model.xgb_model.get_booster()
    dmat = xgb.DMatrix(X_unified, feature_names=model.all_features)
    shap_matrix = booster.predict(dmat, pred_contribs=True)[:, :-1]  # drop bias column

    friendly_names = {
        "Normalized_Hippo": "Hippocampal Volume Ratio",
        "Normalized_Cortex": "Cortical Gray Matter Ratio",
        "Normalized_SubCortGray": "Subcortical Gray Matter Ratio",
        "Normalized_WhiteMatter": "Cerebral White Matter Ratio",
        "Hippo_Asymmetry": "Hemispheric Hippocampal Asymmetry",
        "Age": "Biological Age",
        "Reserve_Masked_Index": "Cognitive Reserve Discordance (Masked Atrophy)",
        "Cognitive_Reserve_Buffer": "Cognitive Reserve Buffer",
        "Vascular_Burden_Index": "Vascular Disease Burden (Stroke/CVD)",
        "Family_Dementia_History": "Familial Genetic Predisposition",
        "Quantum_Entangled_Q0": "Quantum Entanglement Q0 (Structural Geometry)",
        "Quantum_Entangled_Q1": "Quantum Entanglement Q1 (Cognitive-Structural Coupling)",
        "Quantum_Entangled_Q2": "Quantum Entanglement Q2 (Vascular-Atrophy Cross-talk)",
        "Quantum_Entangled_Q3": "Quantum Entanglement Q3 (Multi-Domain Hilbert Feature)",
        "Quantum_Entangled_Q4": "Quantum Entanglement Q4 (Non-linear Progression)",
        "Quantum_Entangled_Q5": "Quantum Entanglement Q5 (Reserve Buffer Projection)"
    }

    # 5. Build Patient-Level Discovery Spreadsheet
    records = []
    for i, row in test_df.iterrows():
        p_risk = float(probs[i])
        shaps = shap_matrix[i]

        # Classify risk tier
        if p_risk >= 0.70:
            tier = "High Impairment Risk"
        elif p_risk >= 0.40:
            tier = "Preclinical Warning Zone"
        else:
            tier = "Low / Resilient Normal"

        # Separate into drivers (+) and protective factors (-)
        feat_shaps = list(zip(model.all_features, shaps))
        drivers = sorted([item for item in feat_shaps if item[1] > 0], key=lambda x: x[1], reverse=True)
        protective = sorted([item for item in feat_shaps if item[1] < 0], key=lambda x: x[1])

        top_driver_name = friendly_names.get(drivers[0][0], drivers[0][0]) if len(drivers) > 0 else "None"
        top_driver_val = float(drivers[0][1]) if len(drivers) > 0 else 0.0

        top_prot_name = friendly_names.get(protective[0][0], protective[0][0]) if len(protective) > 0 else "None"
        top_prot_val = float(protective[0][1]) if len(protective) > 0 else 0.0

        rec = {
            "Subject_ID": row["Subject"],
            "Session": row["Session"],
            "Age": round(float(row["Age"]), 1),
            "Clinical_Status": row["conversion_status"],
            "Ground_Truth_CDR": float(row["cdr_total"]),
            "Ground_Truth_Dx": row["dx1"],
            "Predicted_Risk_Score": round(p_risk * 100, 2),
            "Risk_Category": tier,
            "Primary_Risk_Driver": top_driver_name,
            "Risk_Driver_SHAP": round(top_driver_val, 4),
            "Primary_Protective_Factor": top_prot_name,
            "Protective_Factor_SHAP": round(top_prot_val, 4),
            "Hippo_Volume_mm3": round(float(row["TOTAL_HIPPOCAMPUS_VOLUME"]), 1),
            "Intracranial_Vol_mm3": round(float(row["IntraCranialVol"]), 1),
            "Cognitive_Reserve_Buffer": round(float(row["Cognitive_Reserve_Buffer"]), 2),
            "Vascular_Burden_Index": int(row["Vascular_Burden_Index"]),
            "Family_Dementia_History": int(row["Family_Dementia_History"]),
            "Reserve_Masked_Discordance": round(float(row["Reserve_Masked_Index"]), 3)
        }

        # Add individual SHAP values for each feature
        for feat in model.all_features:
            fname_clean = f"SHAP_{feat}"
            feat_idx = model.all_features.index(feat)
            rec[fname_clean] = round(float(shaps[feat_idx]), 4)

        records.append(rec)

    discovery_df = pd.DataFrame(records)
    out_csv = "reports/cohort_biomarker_discovery.csv"
    os.makedirs("reports", exist_ok=True)
    discovery_df.to_csv(out_csv, index=False)
    print(f"\nSuccessfully generated patient discovery CSV: {out_csv} ({len(discovery_df)} rows)")

    # 6. Global Biomarker Importance Ranking across the entire cohort
    print("\n" + "=" * 80)
    print("GLOBAL NEURODEGENERATION BIOMARKER IMPORTANCE RANKING (Mean |SHAP|)")
    print("=" * 80)

    mean_abs_shaps = np.mean(np.abs(shap_matrix), axis=0)
    global_ranking = pd.DataFrame({
        "Feature": model.all_features,
        "Biomarker_Name": [friendly_names.get(f, f) for f in model.all_features],
        "Mean_Absolute_SHAP": mean_abs_shaps
    }).sort_values("Mean_Absolute_SHAP", ascending=False).reset_index(drop=True)

    for rank, r in global_ranking.iterrows():
        is_q = " [Quantum]" if r["Feature"].startswith("Quantum") else ""
        print(f"  #{rank+1:2d} | {r['Biomarker_Name']:<52} | Impact: {r['Mean_Absolute_SHAP']:.4f}{is_q}")

    # 7. Render Publication-Grade Summary Plot
    plot_path = "reports/global_biomarker_importance.png"
    plt.figure(figsize=(11, 7), dpi=150)
    colors = ["#7b1fa2" if f.startswith("Quantum") else "#1565c0" for f in global_ranking["Feature"]]
    
    bars = plt.barh(global_ranking["Biomarker_Name"][::-1], global_ranking["Mean_Absolute_SHAP"][::-1], color=colors[::-1], edgecolor="black", alpha=0.85)
    plt.xlabel("Mean Absolute SHAP Value (Global Predictive Impact)", fontsize=11, fontweight="bold")
    plt.title("Identified Neurodegeneration Biomarkers Across OASIS-3 Cohort\n(Purple: PennyLane Quantum Features | Blue: Clinical & Imaging Features)", fontsize=13, fontweight="bold")
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    
    # Custom Legend
    import matplotlib.patches as mpatches
    legend_elements = [
        mpatches.Patch(facecolor="#1565c0", edgecolor="black", label="Clinical & Structural MRI Biomarkers"),
        mpatches.Patch(facecolor="#7b1fa2", edgecolor="black", label="PennyLane Quantum Entanglement Features")
    ]
    plt.legend(handles=legend_elements, loc="lower right", fontsize=10)

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved Global Biomarker Ranking Chart: {plot_path}")

    # 8. High-Level Clinical Discovery Insights
    pre_converters = discovery_df[discovery_df["Clinical_Status"] == "pre-conversion"]
    top_driver_preconv = pre_converters["Primary_Risk_Driver"].value_counts()
    print("\n" + "-" * 70)
    print(f"DISCOVERY: Primary Risk Drivers in Pre-Conversion Patients ({len(pre_converters)} scans):")
    for b_name, count in top_driver_preconv.head(3).items():
        print(f"  • {b_name:<46}: Triggered in {count} scans ({count/len(pre_converters):.1%})")

    resilient_controls = discovery_df[(discovery_df["Clinical_Status"] == "stable-normal") & (discovery_df["Age"] >= 75)]
    top_prot_resilient = resilient_controls["Primary_Protective_Factor"].value_counts()
    print(f"\nDISCOVERY: Primary Protective Biomarkers in Resilient Elders (Age >= 75, N={len(resilient_controls)}):")
    for b_name, count in top_prot_resilient.head(3).items():
        print(f"  • {b_name:<46}: Shielding {count} scans ({count/len(resilient_controls):.1%})")
    print("-" * 70)

if __name__ == "__main__":
    main()
