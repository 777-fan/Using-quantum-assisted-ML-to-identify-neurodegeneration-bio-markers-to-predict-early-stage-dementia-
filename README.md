# Using Quantum-Assisted Machine Learning to Identify Neurodegeneration Biomarkers and Predict Early-Stage Dementia Risk

An end-to-end clinical machine learning framework that combines **PennyLane Quantum Computing (Variational Quantum Circuits)**, **Gradient Boosted Decision Trees (XGBoost)**, **TreeSHAP Explainable AI**, and **Monte Carlo Longitudinal Trajectory Simulations** to predict early-stage Alzheimer's disease spectrum dementia risk from brain MRI scans using the OASIS-3 cohort.

---

## Architecture Overview

```
                [ Patient Brain MRI Scan ]
       (Age, Hippo Volume, ICV, Cortex, Subcortical, WM)
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
[ Classical Volumetric Biomarkers ]  [ PennyLane Quantum Circuit ]
• Normalized Hippo                   • 4-Qubit Entangled Circuit
• Normalized Cortex                  • Non-linear geometric mapping
• Subcortical Gray Matter            • Outputs: Quantum Interaction Features
• White Matter & Asymmetry                      (Q0, Q1, Q2, Q3)
            │                                 │
            └────────────────┬────────────────┘
                             ▼
             [ Unified 10D Enriched Feature Space ]
                             │
                             ▼
             [ Single Unified XGBoost Engine ]
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
  ★ ONE UNIFIED RISK SCORE ★        ★ PER-PREDICTION BIOMARKERS ★
   (0.0% to 100.0% Probability)       (TreeSHAP Local Attribution)
                             │
                             ▼
            [ Monte Carlo 10-Year Forward Engine ]
   Stochastic annual atrophy sampling: ΔV ~ N(μ_atrophy, σ_atrophy)
                             │
                             ▼
  ★ ESTIMATED WINDOW TO DIAGNOSIS (IF APPLICABLE) & 95% CI ★
```

---

## 🔬 Key Components

1. **Strict Patient-Level Split (0% Leakage)**:
   - OASIS-3 is a longitudinal cohort. Patients are partitioned strictly by `Subject` ID across training and holdout test sets so that past, present, and future scans of a subject never leak between partitions.
2. **PennyLane Quantum Layer**:
   - Parameterized 4-qubit circuit utilizing angle embedding and circular entangling layers (`CNOT`) to project brain volumetric interactions into a $2^4 = 16$-dimensional Hilbert space.
3. **Unified Gradient Boosted Trees**:
   - Single unified XGBoost classifier trained on classical FreeSurfer metrics and quantum entanglement expectation values.
4. **Per-Prediction Biomarker Attribution (TreeSHAP)**:
   - Every inference identifies the **Primary Disease Risk Driver** and **Primary Protective Biomarker**, along with a full feature waterfall breakdown.
5. **Monte Carlo Longitudinal Trajectory Simulation**:
   - Runs 1,000 stochastic forward simulation paths across a 10-year projection horizon to determine the probability of conversion and the **estimated window of time until clinical diagnosis ($CDR \ge 0.5$)**.

---

## Quickstart

### 1. Installation
Clone the repository and set up a virtual environment:
```bash
git clone git@github.com:777-fan/Using-quantum-assisted-ML-to-identify-neurodegeneration-bio-markers-to-predict-early-stage-dementia-.git
cd Using-quantum-assisted-ML-to-identify-neurodegeneration-bio-markers-to-predict-early-stage-dementia-

python3 -m venv venv
source venv/bin/activate
pip install numpy pandas scikit-learn xgboost torch pennylane "autoray<0.8.0" matplotlib joblib
```

### 2. Predict on a Test Patient
Run prediction and biomarker attribution on an unseen holdout patient:
```bash
python predict.py --patient OAS30341
```

### 3. Predict on Custom Brain MRI Measurements
```bash
python predict.py --age 71.5 --hippo 5800 --icv 1500000 --cortex 410000
```

### 4. Rerun the Complete Pipeline
```bash
python run_pipeline.py
```

---

## 📊 Sample Output
```text
======================================================================
UNIFIED QUANTUM-CLASSICAL DEMENTIA RISK ASSESSMENT
======================================================================
1. UNIFIED DEMENTIA RISK SCORE:
   ► Predicted Risk: 44.2% [PRECLINICAL / ELEVATED RISK (Transition Warning Zone)]

2. IDENTIFYING BIOMARKERS:
   [!] Primary Disease Risk Driver:
       ► Subcortical Gray Matter Loss
         • SHAP Impact: +0.1027 (Elevates dementia risk)
         • Measured Value: 0.0351
   [*] Primary Protective Biomarker:
       ► Hippocampal Volume Atrophy (Normalized to ICV)
         • SHAP Impact: -0.4399 (Buffers against cognitive decline)
         • Measured Value: 0.0044

3. BIOMARKER CONTRIBUTION BREAKDOWN (TreeSHAP Waterfall):
   [+] Risk Factors (Pushing risk higher):
       • Subcortical Gray Matter Loss                  : SHAP = +0.1027
       • Biological Aging                              : SHAP = +0.1012
       • Quantum Entanglement Q3 (Non-linear Anomaly)  : SHAP = +0.0916
   [-] Protective Factors (Preserving cognitive stability):
       • Hippocampal Volume Atrophy (Normalized to ICV): SHAP = -0.4399
       • Cortical Gray Matter Thinning                 : SHAP = -0.1910

4. MONTE CARLO LONGITUDINAL PROJECTION (10-Year Horizon):
   ► 10-Year Conversion Risk Probability: 100.0%
   ► Estimated Window to Diagnosis (if applicable): 0.4 years (95% CI: [0.4 - 1.6] years)

5. TRAJECTORY FAN CHART:
   ► Saved to: reports/prediction_trajectory.png
======================================================================
```
