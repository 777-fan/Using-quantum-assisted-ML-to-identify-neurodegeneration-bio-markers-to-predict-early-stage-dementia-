"""
End-to-End Pipeline Driver:
1. Data Preparation & Strict Patient-Level Holdout Split
2. Unified Quantum-Embedded XGBoost Model Training
3. Unseen Holdout Evaluation (Zero Leakage)
4. Model Artifact Serialization
5. Monte Carlo Forward Trajectory Simulation & Diagnosis Window Estimation
6. Per-Prediction Primary Biomarker Identification & TreeSHAP Waterfall
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
import pandas as pd
from src.prepare_data import prepare_oasis3_data
from src.unified_quantum_model import UnifiedQuantumXGBoost
from src.monte_carlo_engine import MonteCarloTrajectorySimulator

def main():
    print("=" * 80)
    print("UNIFIED QUANTUM-EMBEDDED XGBOOST DEMENTIA RISK & BIOMARKER PIPELINE")
    print("=" * 80)

    # 1. Prepare Data
    train_csv = "data/oasis3_train_patients.csv"
    test_csv = "data/oasis3_holdout_test_patients.csv"

    if not os.path.exists(train_csv) or not os.path.exists(test_csv):
        train_csv, test_csv = prepare_oasis3_data()

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    print(f"Loaded Train Set: {len(train_df)} rows ({train_df['Subject'].nunique()} subjects)")
    print(f"Loaded Holdout Test Set: {len(test_df)} rows ({test_df['Subject'].nunique()} subjects)")

    # 2. Train Single Unified Model
    unified_model = UnifiedQuantumXGBoost(random_state=42)
    unified_model.fit(train_df)

    # 3. Evaluate on Unseen Holdout Cohort
    eval_results = unified_model.evaluate(test_df, name="Holdout Test Set (Unseen Patients)")

    # 4. Save Model Artifacts
    unified_model.save(save_dir="saved_models")

    # 5. Monte Carlo Longitudinal Simulation & Biomarker Identification on Holdout Patients
    print("\n" + "=" * 80)
    print("RUNNING MONTE CARLO TRAJECTORY SIMULATION & BIOMARKER IDENTIFICATION")
    print("=" * 80)

    mc_engine = MonteCarloTrajectorySimulator(
        unified_model,
        threshold=0.50,
        num_simulations=1000,
        horizon_years=10,
        random_seed=42
    )

    demo_subjects = [
        ("OAS30341", "Pre-Conversion", "reports/monte_carlo_trajectory_converter.png"),
        ("OAS31222", "Borderline / Intermediate", "reports/monte_carlo_trajectory_intermediate.png"),
        ("OAS31133", "Stable Normal", "reports/monte_carlo_trajectory_normal.png")
    ]

    for subj_id, desc, plot_path in demo_subjects:
        matches = test_df[test_df["Subject"] == subj_id]
        if len(matches) == 0:
            continue
        patient_row = matches.iloc[0]

        print(f"\n" + "-" * 70)
        print(f"PATIENT EVALUATION: {subj_id} ({desc})")
        print(f"  Age: {patient_row['Age']:.1f} y/o | Hippo: {patient_row['TOTAL_HIPPOCAMPUS_VOLUME']:.1f} mm³ | ICV: {patient_row['IntraCranialVol']:.1f} mm³")
        print(f"  Ground Truth: CDR={patient_row['cdr_total']}, Dx={patient_row['dx1']}, Status={patient_row['conversion_status']}")

        res = mc_engine.simulate_patient(patient_row)
        mc_engine.plot_trajectory(res, patient_id=f"Subject {subj_id} ({desc})", save_path=plot_path)

        exp = res["explanation"]
        print(f"  ► Unified Baseline Risk: {res['p_base']:.1%}")

        if exp is not None:
            pb = exp["primary_biomarker"]
            print(f"  ► PRIMARY IDENTIFYING BIOMARKER: {pb['friendly_name']}")
            print(f"     • SHAP Contribution: {pb['shap_value']:+.4f}")
            if len(exp["risk_drivers"]) > 0:
                print(f"     • Top Risk Driver:  {exp['risk_drivers'][0]['friendly_name']} ({exp['risk_drivers'][0]['shap_value']:+.4f})")
            if len(exp["protective_factors"]) > 0:
                print(f"     • Top Protective:   {exp['protective_factors'][0]['friendly_name']} ({exp['protective_factors'][0]['shap_value']:+.4f})")

        print(f"  ► 10-Year Conversion Risk Probability: {res['conversion_prob_10yr']:.1%}")
        if res["median_time_to_diagnosis"] is not None:
            med = res["median_time_to_diagnosis"]
            ci = res["ci_95"]
            print(f"  ► Estimated Window to Diagnosis (if applicable): {med:.1f} years (95% CI: [{ci[0]:.1f} - {ci[1]:.1f}] years)")
        else:
            print("  ► Estimated Window to Diagnosis (if applicable): N/A (Projected Trajectory safely remains below threshold across 10 years)")

    print("\n" + "=" * 80)
    print("UNIFIED MODEL PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
