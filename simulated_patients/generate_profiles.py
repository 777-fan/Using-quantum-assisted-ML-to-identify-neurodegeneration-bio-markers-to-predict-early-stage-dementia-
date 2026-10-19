"""
Simulated Patient Profile Generator & Catalog.
Generates realistic clinical JSON profiles for testing:
1. patient_vascular_accelerated.json (Pre-converter with stroke/CVD accelerating decline)
2. patient_cognitive_reserve_masked.json (Hippocampal loss masked by high education/cognitive buffer)
3. patient_resilient_healthy_control.json (Robust brain volume, zero vascular risk)
"""

import os
import json

PATIENTS = {
    "patient_vascular_accelerated.json": {
        "Description": "Preclinical 72 y/o with moderate hippocampal loss plus hypertension & stroke history",
        "Age": 72.0,
        "TOTAL_HIPPOCAMPUS_VOLUME": 5950.0,
        "IntraCranialVol": 1460000.0,
        "CortexVol": 405000.0,
        "SubCortGrayVol": 50500.0,
        "CorticalWhiteMatterVol": 410000.0,
        "Hippo_Asymmetry": 0.08,
        "Cognitive_Reserve_Buffer": 0.88,
        "Vascular_Burden_Index": 2,
        "Family_Dementia_History": 1
    },
    "patient_cognitive_reserve_masked.json": {
        "Description": "69 y/o with significant hippocampal shrinkage (5,500 mm3) but masked by high cognitive buffer (MMSE 29/30)",
        "Age": 69.5,
        "TOTAL_HIPPOCAMPUS_VOLUME": 5500.0,
        "IntraCranialVol": 1420000.0,
        "CortexVol": 418000.0,
        "SubCortGrayVol": 52000.0,
        "CorticalWhiteMatterVol": 425000.0,
        "Hippo_Asymmetry": 0.04,
        "Cognitive_Reserve_Buffer": 0.97,
        "Vascular_Burden_Index": 0,
        "Family_Dementia_History": 1
    },
    "patient_resilient_healthy_control.json": {
        "Description": "71 y/o resilient control with robust hippocampal volume (8,200 mm3) and zero vascular risk",
        "Age": 71.0,
        "TOTAL_HIPPOCAMPUS_VOLUME": 8200.0,
        "IntraCranialVol": 1600000.0,
        "CortexVol": 475000.0,
        "SubCortGrayVol": 57000.0,
        "CorticalWhiteMatterVol": 460000.0,
        "Hippo_Asymmetry": 0.02,
        "Cognitive_Reserve_Buffer": 1.0,
        "Vascular_Burden_Index": 0,
        "Family_Dementia_History": 0
    }
}

def generate_profiles(target_dir="simulated_patients"):
    os.makedirs(target_dir, exist_ok=True)
    for filename, profile in PATIENTS.items():
        filepath = os.path.join(target_dir, filename)
        with open(filepath, "w") as f:
            json.dump(profile, f, indent=2)
        print(f"Generated: {filepath}")

if __name__ == "__main__":
    generate_profiles()
