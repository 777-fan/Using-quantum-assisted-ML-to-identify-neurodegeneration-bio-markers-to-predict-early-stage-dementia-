"""
End-to-End Multi-Modal Pipeline Driver:
1. Multi-Modal Dataset Building & Strict Subject-Level Split
2. Unified 6-Qubit PennyLane Quantum + XGBoost Multi-Modal Model Training
3. Unseen Holdout Evaluation (Zero Leakage)
4. Model Artifact Serialization
5. Direction #4 Dynamic Sigmoidal Biological Cascade Simulation & Multi-Domain TreeSHAP
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
import pandas as pd
from src.build_multimodal_dataset import build_multimodal_dataset
from src.unified_quantum_model import UnifiedMultiModalQuantumModel
from src.monte_carlo_engine import DynamicCascadeSimulator

def main():
    print("=" * 80)
    print("MULTI-MODAL QUANTUM-EMBEDDED DEMENTIA RISK & BIOMARKER CASCADE PIPELINE")
    print("=" * 80)

    train_csv = "data/oasis3_multimodal_train.csv"
    test_csv = "data/oasis3_multimodal_test.csv"

    if not os.path.exists(train_csv) or not os.path.exists(test_csv):
        train_csv, test_csv = build_multimodal_dataset()

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    print(f"Loaded Multi-Modal Train Set: {len(train_df)} rows ({train_df['Subject'].nunique()} subjects)")
    print(f"Loaded Multi-Modal Test Set:  {len(test_df)} rows ({test_df['Subject'].nunique()} subjects)")

    # Train Unified 6-Qubit Model
    model = UnifiedMultiModalQuantumModel(random_state=42)
    model.fit(train_df)

    # Evaluate on Holdout Cohort
    model.evaluate(test_df, name="Holdout Multi-Modal Cohort (Zero Leakage)")

    # Save Artifacts
    model.save(save_dir="saved_models")

    # Direction #4 Dynamic Sigmoidal Simulation
    print("\n" + "=" * 80)
    print("RUNNING DIRECTION #4 SIGMOIDAL BIOLOGICAL CASCADE SIMULATION")
    print("=" * 80)

    sim = DynamicCascadeSimulator(model, threshold=0.50, num_simulations=1000, horizon_years=10)

    demo_subjects = [
        ("OAS30341", "Pre-Conversion", "reports/monte_carlo_trajectory_converter.png"),
        ("OAS31222", "Borderline / Vascular Impaired", "reports/monte_carlo_trajectory_intermediate.png"),
        ("OAS31133", "Resilient Stable Normal", "reports/monte_carlo_trajectory_normal.png")
    ]

    for subj_id, desc, plot_path in demo_subjects:
        matches = test_df[test_df["Subject"] == subj_id]
        if len(matches) == 0:
            continue
        row = matches.iloc[0]

        print(f"\n" + "-" * 70)
        print(f"EVALUATING: Subject {subj_id} ({desc})")
        print(f"  Age: {row['Age']:.1f} | Hippo: {row['TOTAL_HIPPOCAMPUS_VOLUME']:.1f} mm³ | ICV: {row['IntraCranialVol']:.1f} mm³")
        print(f"  Vascular Burden Index: {row['Vascular_Burden_Index']} | Family History: {row['Family_Dementia_History']}")
        print(f"  Reserve Masked Discordance: {row['Reserve_Masked_Index']:.3f}")

        res = sim.simulate_patient(row)
        sim.plot_trajectory(res, patient_id=f"Subject {subj_id} ({desc})", save_path=plot_path)

        exp = res["explanation"]
        print(f"  ► Unified Multi-Modal Risk: {res['p_base']:.1%}")
        print(f"  ► Personalized Biological Atrophy Velocity: {res['mu_atrophy_annual']:.2f}%/year (Acceleration: {res['vasc_acceleration_multiplier']:.2f}x)")

        if exp["primary_risk_driver"]:
            prd = exp["primary_risk_driver"]
            print(f"  ► Primary Risk Driver:  {prd['friendly_name']} (SHAP = {prd['shap_value']:+.4f})")
        if exp["primary_protective_biomarker"]:
            ppb = exp["primary_protective_biomarker"]
            print(f"  ► Primary Protective:   {ppb['friendly_name']} (SHAP = {ppb['shap_value']:+.4f})")

        print(f"  ► 10-Year Conversion Risk Probability: {res['conversion_prob_10yr']:.1%}")
        if res["median_time_to_diagnosis"] is not None:
            med = res["median_time_to_diagnosis"]
            ci = res["ci_95"]
            print(f"  ► Estimated Window to Diagnosis (if applicable): {med:.1f} years (95% CI: [{ci[0]:.1f} - {ci[1]:.1f}] years)")
        else:
            print("  ► Estimated Window to Diagnosis (if applicable): N/A (Projected Trajectory safely remains below threshold across 10 years)")

    print("\n" + "=" * 80)
    print("MULTI-MODAL PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
