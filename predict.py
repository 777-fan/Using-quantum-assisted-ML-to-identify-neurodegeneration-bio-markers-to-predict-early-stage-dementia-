"""
Multi-Modal Inference & Direction #4 Dynamic Sigmoidal Simulation CLI Tool.
Usage:
  # 1. Predict on a holdout patient ID:
  ./venv/bin/python3 predict.py --patient OAS30341

  # 2. Predict on a simulated patient JSON profile:
  ./venv/bin/python3 predict.py --file simulated_patients/patient_vascular_accelerated.json
  ./venv/bin/python3 predict.py --file simulated_patients/patient_cognitive_reserve_masked.json
  ./venv/bin/python3 predict.py --file simulated_patients/patient_resilient_healthy_control.json

  # 3. Predict on custom terminal arguments:
  ./venv/bin/python3 predict.py --age 71.5 --hippo 5800 --icv 1450000 --cortex 405000 --vascular 2 --family 1
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
import json
import argparse
import pandas as pd
import numpy as np
from src.unified_quantum_model import UnifiedMultiModalQuantumModel
from src.monte_carlo_engine import DynamicCascadeSimulator

def main():
    parser = argparse.ArgumentParser(description="Multi-Modal Quantum Dementia Risk & Dynamic Sigmoidal Cascade Engine.")
    parser.add_argument("--patient", type=str, help="Subject ID from holdout set (e.g. OAS30341, OAS31222, OAS31133)")
    parser.add_argument("--file", type=str, help="Path to simulated patient JSON file (e.g. simulated_patients/patient_vascular_accelerated.json)")
    parser.add_argument("--age", type=float, default=71.0, help="Patient age (years)")
    parser.add_argument("--hippo", type=float, default=6200.0, help="Total hippocampal volume (mm3)")
    parser.add_argument("--icv", type=float, default=1500000.0, help="Intracranial volume (mm3)")
    parser.add_argument("--cortex", type=float, default=420000.0, help="Cortex volume (mm3)")
    parser.add_argument("--vascular", type=int, default=0, help="Vascular burden index (0 to 3: stroke, hypertension, heart disease)")
    parser.add_argument("--family", type=int, default=0, help="Family dementia history (1 = parental dementia, 0 = none)")
    parser.add_argument("--reserve", type=float, default=0.95, help="Cognitive reserve buffer (MMSE / 30, e.g. 0.95 = 28.5/30)")
    parser.add_argument("--sims", type=int, default=1000, help="Number of Monte Carlo simulations")
    parser.add_argument("--out", type=str, default="reports/prediction_trajectory.png", help="Output plot path")

    args = parser.parse_args()

    model = UnifiedMultiModalQuantumModel.load("saved_models")
    sim = DynamicCascadeSimulator(model, threshold=0.50, num_simulations=args.sims, horizon_years=10)

    if args.file:
        with open(args.file, "r") as f:
            data = json.load(f)
        patient_row = pd.Series(data)
        label = f"Simulated Profile: {os.path.basename(args.file)}"
        print(f"\nEvaluating Simulated Patient Profile from: {args.file}")
        if "Description" in data:
            print(f"  Description: {data['Description']}")
    elif args.patient:
        test_df = pd.read_csv("data/oasis3_multimodal_test.csv")
        matches = test_df[test_df["Subject"] == args.patient]
        if len(matches) == 0:
            print(f"Error: Subject '{args.patient}' not found in data/oasis3_multimodal_test.csv")
            return
        patient_row = matches.iloc[0]
        label = f"Subject {args.patient}"
        print(f"\nEvaluating Holdout Subject: {args.patient}")
        print(f"  Clinical Ground Truth: CDR={patient_row['cdr_total']}, Dx={patient_row['dx1']}, Status={patient_row['conversion_status']}")
    else:
        # Build from CLI arguments
        patient_row = pd.Series({
            "Age": args.age,
            "TOTAL_HIPPOCAMPUS_VOLUME": args.hippo,
            "IntraCranialVol": args.icv,
            "Normalized_Cortex": args.cortex / args.icv,
            "Normalized_SubCortGray": 52000.0 / args.icv,
            "Normalized_WhiteMatter": 420000.0 / args.icv,
            "Hippo_Asymmetry": 0.05,
            "Cognitive_Reserve_Buffer": args.reserve,
            "Vascular_Burden_Index": args.vascular,
            "Family_Dementia_History": args.family
        })
        label = f"Custom Patient (Age {args.age:.0f}, Hippo {args.hippo:.0f} mm³)"
        print(f"\nEvaluating Custom Patient Inputs:")

    print(f"  Age: {patient_row['Age']:.1f} y/o | Hippo Vol: {patient_row['TOTAL_HIPPOCAMPUS_VOLUME']:.1f} mm³ | ICV: {patient_row['IntraCranialVol']:.1f} mm³")
    print(f"  Vascular Burden Index: {patient_row.get('Vascular_Burden_Index', 0)} | Family History: {patient_row.get('Family_Dementia_History', 0)} | Reserve Buffer: {patient_row.get('Cognitive_Reserve_Buffer', 0.95):.2f}")

    results = sim.simulate_patient(patient_row)
    plot_file = sim.plot_trajectory(results, patient_id=label, save_path=args.out)

    prob = results["p_base"]
    exp = results["explanation"]

    if prob >= 0.70:
        risk_tier = "HIGH RISK (Active Clinical Neurodegenerative Pattern)"
    elif prob >= 0.40:
        risk_tier = "PRECLINICAL RISK (Transition Warning Zone - Intervention Window Open)"
    else:
        risk_tier = "LOW RISK (Within Healthy Resilient Normative Range)"

    print("\n" + "=" * 70)
    print("MULTI-MODAL QUANTUM-ASSISTED DEMENTIA RISK ASSESSMENT")
    print("=" * 70)
    print(f"1. UNIFIED MULTI-MODAL RISK SCORE:")
    print(f"   ► Predicted Risk: {prob:.1%} [{risk_tier}]")
    print(f"   ► Personalized Atrophy Velocity: {results['mu_atrophy_annual']:.2f}%/year (Vascular Multiplier: {results['vasc_acceleration_multiplier']:.2f}x)")

    print(f"\n2. MULTI-DOMAIN IDENTIFYING BIOMARKERS:")
    if exp["primary_risk_driver"]:
        prd = exp["primary_risk_driver"]
        print(f"   [!] Primary Disease Risk Driver:")
        print(f"       ► {prd['friendly_name']}")
        print(f"         • SHAP Impact: {prd['shap_value']:+.4f} (Accelerates Dementia Trajectory)")
        print(f"         • Measured Feature Value: {prd['measured_value']:.4f}")
    if exp["primary_protective_biomarker"]:
        ppb = exp["primary_protective_biomarker"]
        print(f"   [*] Primary Protective Biomarker:")
        print(f"       ► {ppb['friendly_name']}")
        print(f"         • SHAP Impact: {ppb['shap_value']:+.4f} (Buffers Against Clinical Decline)")
        print(f"         • Measured Feature Value: {ppb['measured_value']:.4f}")

    print(f"\n3. BIOMARKER CONTRIBUTION BREAKDOWN (TreeSHAP Waterfall):")
    if len(exp["risk_drivers"]) > 0:
        print("   [+] Risk Factors (Pushing risk higher):")
        for item in exp["risk_drivers"][:4]:
            print(f"       • {item['friendly_name']:<52}: SHAP = {item['shap_value']:+.4f}")
    if len(exp["protective_factors"]) > 0:
        print("   [-] Protective Factors (Preserving cognitive stability):")
        for item in exp["protective_factors"][:4]:
            print(f"       • {item['friendly_name']:<52}: SHAP = {item['shap_value']:+.4f}")

    print(f"\n4. DIRECTION #4 SIGMOIDAL BIOLOGICAL CASCADE PROJECTION (10-Year Horizon):")
    print(f"   ► 10-Year Conversion Risk Probability:          {results['conversion_prob_10yr']:.1%}")
    if results["median_time_to_diagnosis"] is not None:
        med = results["median_time_to_diagnosis"]
        ci = results["ci_95"]
        print(f"   ► Estimated Window to Diagnosis (if applicable): {med:.1f} years")
        print(f"   ► 95% Confidence Interval:                       [{ci[0]:.1f} - {ci[1]:.1f}] years")
    else:
        print("   ► Estimated Window to Diagnosis (if applicable): N/A (Safely remains below 50% threshold)")

    print(f"\n5. TRAJECTORY FAN CHART:")
    print(f"   ► Saved to: {plot_file}")
    print("=" * 70)

if __name__ == "__main__":
    main()
