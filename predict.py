"""
Unified Quantum-XGBoost Prediction & Biomarker Identification CLI Tool.
Usage:
  # Predict on a holdout patient ID:
  ./venv/bin/python3 predict.py --patient OAS30341
  ./venv/bin/python3 predict.py --patient OAS31222
  ./venv/bin/python3 predict.py --patient OAS31133

  # Predict on custom MRI numbers:
  ./venv/bin/python3 predict.py --age 71.5 --hippo 5800 --icv 1500000 --cortex 410000
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
import argparse
import pandas as pd
import numpy as np
from src.unified_quantum_model import UnifiedQuantumXGBoost
from src.monte_carlo_engine import MonteCarloTrajectorySimulator

def main():
    parser = argparse.ArgumentParser(description="Predict dementia risk, identify driving biomarkers, and estimate diagnosis window.")
    parser.add_argument("--patient", type=str, help="Subject ID from holdout test set (e.g. OAS30341, OAS31222, OAS31133)")
    parser.add_argument("--age", type=float, default=70.0, help="Patient age (years)")
    parser.add_argument("--hippo", type=float, default=6500.0, help="Total hippocampal volume (mm3)")
    parser.add_argument("--icv", type=float, default=1500000.0, help="Intracranial volume (mm3)")
    parser.add_argument("--cortex", type=float, default=420000.0, help="Cortex volume (mm3)")
    parser.add_argument("--sims", type=int, default=1000, help="Number of Monte Carlo simulations")
    parser.add_argument("--out", type=str, default="reports/prediction_trajectory.png", help="Output plot path")

    args = parser.parse_args()

    # Load unified model
    model = UnifiedQuantumXGBoost.load("saved_models")
    mc = MonteCarloTrajectorySimulator(model, threshold=0.50, num_simulations=args.sims)

    if args.patient:
        test_df = pd.read_csv("data/oasis3_holdout_test_patients.csv")
        matches = test_df[test_df["Subject"] == args.patient]
        if len(matches) == 0:
            print(f"Error: Subject '{args.patient}' not found in data/oasis3_holdout_test_patients.csv")
            return
        patient_row = matches.iloc[0]
        label = f"Subject {args.patient}"
        print(f"\nEvaluating Holdout Subject: {args.patient}")
        print(f"  Clinical Ground Truth: CDR={patient_row['cdr_total']}, Dx={patient_row['dx1']}, Status={patient_row['conversion_status']}")
    else:
        # Construct synthetic patient row
        patient_row = pd.Series({
            "Age": args.age,
            "TOTAL_HIPPOCAMPUS_VOLUME": args.hippo,
            "IntraCranialVol": args.icv,
            "Normalized_Cortex": args.cortex / args.icv,
            "Normalized_SubCortGray": 53000.0 / args.icv,
            "Normalized_WhiteMatter": 420000.0 / args.icv,
            "Hippo_Asymmetry": 0.05,
        })
        label = f"Custom Patient (Age {args.age:.0f}, Hippo {args.hippo:.0f} mm³)"
        print(f"\nEvaluating Custom Brain Scan Inputs:")

    print(f"  Age: {patient_row['Age']:.1f} y/o")
    print(f"  Hippocampal Volume: {patient_row['TOTAL_HIPPOCAMPUS_VOLUME']:.1f} mm³")
    print(f"  Intracranial Volume (ICV): {patient_row['IntraCranialVol']:.1f} mm³")

    # Run Monte Carlo Forward Simulation with the Unified Model
    results = mc.simulate_patient(patient_row)
    plot_file = mc.plot_trajectory(results, patient_id=label, save_path=args.out)

    explanation = results["explanation"]
    prob = results["p_base"]

    if prob >= 0.70:
        risk_tier = "HIGH RISK (Severe Impairment / Dementia Pattern)"
    elif prob >= 0.40:
        risk_tier = "PRECLINICAL / ELEVATED RISK (Transition Warning Zone)"
    else:
        risk_tier = "LOW RISK (Within Healthy Normative Range)"

    print("\n" + "=" * 70)
    print("UNIFIED QUANTUM-CLASSICAL DEMENTIA RISK ASSESSMENT")
    print("=" * 70)
    print(f"1. UNIFIED DEMENTIA RISK SCORE:")
    print(f"   ► Predicted Risk: {prob:.1%} [{risk_tier}]")

    if explanation is not None:
        prd = explanation.get("primary_risk_driver")
        ppb = explanation.get("primary_protective_biomarker")

        print(f"\n2. IDENTIFYING BIOMARKERS:")
        if prd:
            print(f"   [!] Primary Disease Risk Driver:")
            print(f"       ► {prd['friendly_name']}")
            print(f"         • SHAP Impact: {prd['shap_value']:+.4f} (Elevates dementia risk)")
            print(f"         • Measured Value: {prd['measured_value']:.4f}")
        if ppb:
            print(f"   [*] Primary Protective Biomarker:")
            print(f"       ► {ppb['friendly_name']}")
            print(f"         • SHAP Impact: {ppb['shap_value']:+.4f} (Buffers against cognitive decline)")
            print(f"         • Measured Value: {ppb['measured_value']:.4f}")

        print(f"\n3. BIOMARKER CONTRIBUTION BREAKDOWN (TreeSHAP Waterfall):")
        if len(explanation["risk_drivers"]) > 0:
            print("   [+] Risk Factors (Pushing risk higher):")
            for item in explanation["risk_drivers"][:4]:
                print(f"       • {item['friendly_name']:<46}: SHAP = {item['shap_value']:+.4f}")
        if len(explanation["protective_factors"]) > 0:
            print("   [-] Protective Factors (Preserving cognitive stability):")
            for item in explanation["protective_factors"][:4]:
                print(f"       • {item['friendly_name']:<46}: SHAP = {item['shap_value']:+.4f}")

    print("\n4. MONTE CARLO LONGITUDINAL PROJECTION (10-Year Horizon):")
    print(f"   ► 10-Year Conversion Risk Probability: {results['conversion_prob_10yr']:.1%}")
    if results["median_time_to_diagnosis"] is not None:
        med = results["median_time_to_diagnosis"]
        ci = results["ci_95"]
        print(f"   ► Estimated Window to Diagnosis (if applicable): {med:.1f} years")
        print(f"   ► 95% Confidence Interval:                      [{ci[0]:.1f} - {ci[1]:.1f}] years")
    else:
        print("   ► Estimated Window to Diagnosis (if applicable): N/A (Safely remains below 50% threshold)")

    print(f"\n5. TRAJECTORY FAN CHART:")
    print(f"   ► Saved to: {plot_file}")
    print("=" * 70)

if __name__ == "__main__":
    main()
