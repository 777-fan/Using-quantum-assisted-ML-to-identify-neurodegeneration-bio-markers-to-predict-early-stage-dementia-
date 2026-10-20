# Using Quantum-Assisted ML to Identify Neurodegeneration Biomarkers & Predict Early-Stage Dementia Risk

An end-to-end multi-modal clinical machine learning and quantum computing system that predicts early-stage Alzheimer's disease spectrum dementia risk from brain MRI scans and clinical history using the OASIS-3 cohort.

The architecture combines **PennyLane 6-Qubit Variational Quantum Circuits**, **Gradient Boosted Decision Trees (XGBoost)**, **TreeSHAP Explainable AI**, and a **Non-Linear Sigmoidal Biological Cascade Monte Carlo Engine** (Jack et al. model) to identify subtle neurodegeneration biomarkers, quantify cognitive reserve buffering, and project personalized 10-year diagnosis time windows.

---

## 📌 Multi-Modal Architecture Overview

```
                       [ MULTI-MODAL CLINICAL REPOSITORIES ]
      ┌─────────────────────────────┼─────────────────────────────┐
      ▼                             ▼                             ▼
[ FreeSurfer Structural MRI ]   [ NACC Form C1 Psychometrics ]  [ Forms A3 & A5 Health/Genetics ]
• Hippocampus Volume (Norm)     • MMSE & Cognitive Buffer       • Vascular Burden Index (Stroke/CVD)
• Cortical Gray Matter Ratio    • Reserve Masked Discordance    • Hypertension & Heart History
• Subcortical & White Matter    • Hemispheric Asymmetry         • Parental Dementia History
      │                             │                             │
      └─────────────────────────────┼─────────────────────────────┘
                                    ▼
                [ Step 1: Multi-Modal Dataset Fusion ]
                • Clean join on `Subject` ID + nearest visit date
                • Strict Patient-Level Split (0% Leakage)
                                    │
                                    ▼
                [ Step 2: 6-Qubit PennyLane Quantum Layer ]
                • Parameterized Quantum Circuit with Cyclic CNOT Entanglement
                • Projects non-linear multi-region & cognitive cross-talk into Hilbert space
                • Outputs: Quantum Entanglement Features (Q0 through Q5)
                                    │
                                    ▼
                [ Step 3: Unified Multi-Modal XGBoost Engine ]
                • Single Gradient Boosted Classifier trained on 16D unified features
                • Output: Unified Multi-Modal Dementia Risk Probability (0.0% to 100.0%)
                                    │
                                    ▼
                [ Step 4: Multi-Domain TreeSHAP Explainer ]
                • Primary Disease Risk Driver (e.g. Medial Temporal Atrophy, Vascular Strain)
                • Primary Protective Biomarker (e.g. Cognitive Reserve Buffer, Preserved Cortex)
                • Full Feature Waterfall Breakdown
                                    │
                                    ▼
            [ Step 5: Dynamic Sigmoidal Biological Cascade (Monte Carlo) ]
            • NON-LINEAR 10-Year Monte Carlo Trajectory (1,000 Stochastic Paths):
              - Early Phase: Cognitive reserve buffers tissue loss (silent phase).
              - Acceleration Phase: Vascular & genetic burden steepens the atrophy slope.
              - Saturation Phase: Late-stage clinical plateau.
                                    │
                                    ▼
   ★ OUTPUT: 
   1. Unified Multi-Modal Dementia Risk Score (%)
   2. Multi-Domain Identifying Biomarkers (Structural + Cognitive + Vascular)
   3. Personalized Biological Atrophy Velocity (%/year tailored by vascular burden)
   4. Estimated Window to Diagnosis (if applicable) & 95% Confidence Interval
   5. Dynamic Sigmoidal Trajectory Fan Chart (PNG)
```

---

## 🔬 Scientific Highlights & Innovations

1. **Zero Patient-Level Data Leakage**:
   - Because OASIS-3 is a longitudinal cohort with repeat scans over decades, subjects are split strictly by **`Subject` ID** (1,710 training scans across 958 subjects vs. 442 holdout test scans across 240 subjects). Past, present, and future scans of a subject never cross between splits.
2. **6-Qubit Multi-Modal Quantum Entanglement**:
   - Compresses structural volumes, cognitive reserve, and vascular burden through a parameterized PennyLane quantum circuit. Entanglement gates (`CNOT`) capture high-dimensional non-linear interactions across tissue loss and clinical risk.
3. **Cognitive Reserve Masked Discordance Index**:
   - Identifies cases where severe hippocampal loss is masked by high education or cognitive baseline (`Hippo_Z` atrophy vs. high MMSE buffer).
4. **Direction #4: Personalized Sigmoidal Biological Cascade**:
   - Instead of simplistic linear atrophy, our Monte Carlo engine models the biological sigmoidal cascade. Patients with stroke/hypertension receive a **vascular multiplier** accelerating annual tissue loss (up to 3.08%/year vs. 1.07%/year for controls).
5. **Multi-Domain TreeSHAP Explainability**:
   - Every inference identifies the exact anatomical or clinical driver pushing risk up, and the protective factors keeping the patient stable.

---

## 📈 Benchmark Performance (Independent Holdout Cohort)

| Model Architecture | Features | ROC-AUC | Accuracy | F1-Score |
| :--- | :--- | :---: | :---: | :---: |
| Classical Baseline | MRI Volumes Only | 0.7751 | 78.05% | 0.5174 |
| Quantum Hybrid Ensemble | MRI Volumes + 4 Qubits | 0.7806 | 77.83% | 0.5243 |
| **Unified Multi-Modal Quantum Model** | **MRI + UDS Health + Reserve + 6 Qubits** | **0.7978** | **81.90%** | **0.5745** |

---

## 🚀 Quickstart & Usage

### 1. Installation
```bash
git clone git@github.com:777-fan/Using-quantum-assisted-ML-to-identify-neurodegeneration-bio-markers-to-predict-early-stage-dementia-.git
cd Using-quantum-assisted-ML-to-identify-neurodegeneration-bio-markers-to-predict-early-stage-dementia-

python3 -m venv venv
source venv/bin/activate
pip install numpy pandas scikit-learn xgboost torch pennylane "autoray<0.8.0" matplotlib joblib
```

### 2. Predict on a Simulated Patient Profile (`simulated_patients/`)
Test realistic clinical archetypes included in the repository:
```bash
# Patient with moderate hippocampal shrinkage + stroke/hypertension history (High Risk)
python predict.py --file simulated_patients/patient_vascular_accelerated.json

# Patient with hippocampal shrinkage masked by high cognitive reserve
python predict.py --file simulated_patients/patient_cognitive_reserve_masked.json

# Resilient control with robust brain volumes and zero vascular burden (Low Risk)
python predict.py --file simulated_patients/patient_resilient_healthy_control.json
```

### 3. Predict on a Real Unseen OASIS-3 Holdout Subject
```bash
python predict.py --patient OAS30341
```

### 4. Predict on Custom Brain MRI & Clinical Inputs
```bash
python predict.py --age 72.0 --hippo 5800 --icv 1460000 --cortex 405000 --vascular 2 --family 1
```

### 5. Retrain the Pipeline from Scratch
```bash
python run_pipeline.py
```

---

## 📊 Sample Inference Output

```text
Evaluating Simulated Patient Profile from: simulated_patients/patient_vascular_accelerated.json
  Description: Preclinical 72 y/o with moderate hippocampal loss plus hypertension & stroke history
  Age: 72.0 y/o | Hippo Vol: 5950.0 mm³ | ICV: 1460000.0 mm³
  Vascular Burden Index: 2 | Family History: 1 | Reserve Buffer: 0.88

======================================================================
MULTI-MODAL QUANTUM-ASSISTED DEMENTIA RISK ASSESSMENT
======================================================================
1. UNIFIED MULTI-MODAL RISK SCORE:
   ► Predicted Risk: 71.3% [HIGH RISK (Active Clinical Neurodegenerative Pattern)]
   ► Personalized Atrophy Velocity: 3.08%/year (Vascular Multiplier: 1.46x)

2. MULTI-DOMAIN IDENTIFYING BIOMARKERS:
   [!] Primary Disease Risk Driver:
       ► Cognitive Reserve Buffer (Baseline Function)
         • SHAP Impact: +0.9814 (Accelerates Dementia Trajectory)
         • Measured Feature Value: 0.8800
   [*] Primary Protective Biomarker:
       ► Cortical Gray Matter Thinning
         • SHAP Impact: -0.7161 (Buffers Against Clinical Decline)
         • Measured Feature Value: 0.2774

3. BIOMARKER CONTRIBUTION BREAKDOWN (TreeSHAP Waterfall):
   [+] Risk Factors (Pushing risk higher):
       • Cognitive Reserve Buffer (Baseline Function)        : SHAP = +0.9814
       • Hippocampal Volume Ratio (Medial Temporal Atrophy)  : SHAP = +0.3967
       • Cognitive Reserve Discordance (Masked Atrophy)      : SHAP = +0.1428
       • Quantum Entanglement Q5 (Reserve Buffer Projection) : SHAP = +0.0952
   [-] Protective Factors (Preserving cognitive stability):
       • Cortical Gray Matter Thinning                       : SHAP = -0.7161
       • Hemispheric Hippocampal Asymmetry                   : SHAP = -0.0859
       • Quantum Entanglement Q2 (Vascular-Atrophy Cross-talk): SHAP = -0.0673

4. DIRECTION #4 SIGMOIDAL BIOLOGICAL CASCADE PROJECTION (10-Year Horizon):
   ► 10-Year Conversion Risk Probability:          100.0%
   ► Estimated Window to Diagnosis (if applicable): 0.0 years (95% CI: [0.0 - 0.0] years)

5. TRAJECTORY FAN CHART:
   ► Saved to: reports/prediction_trajectory.png
======================================================================
```

---

## 📁 Repository Structure

```text
├── predict.py                     <-- Multi-modal CLI tool (IDs, JSON profiles, custom inputs)
├── run_pipeline.py                <-- Master training and evaluation driver
├── README.md                      <-- Project documentation & architecture
├── .gitignore
│
├── src/
│   ├── build_multimodal_dataset.py <-- Fuses FreeSurfer MRI with UDS Form A5 & A3
│   ├── unified_quantum_model.py    <-- 6-qubit PennyLane VQC + XGBoost + TreeSHAP
│   ├── monte_carlo_engine.py       <-- Direction #4 Sigmoidal Biological Cascade Simulator
│   ├── prepare_data.py             <-- MRI-only dataset partitioner
│   └── quantum_xgboost_model.py    <-- 4-qubit baseline model
│
├── simulated_patients/            <-- Pre-configured clinical JSON profiles
│   ├── patient_vascular_accelerated.json
│   ├── patient_cognitive_reserve_masked.json
│   ├── patient_resilient_healthy_control.json
│   └── generate_profiles.py
│
├── saved_models/                  <-- Pre-trained weights and preprocessors
│   ├── multimodal_xgboost_model.json
│   ├── multimodal_preprocessor.joblib
│   ├── unified_xgboost_model.json
│   └── unified_preprocessor.joblib
│
├── data/                          <-- Partitioned train/test datasets
│   ├── oasis3_multimodal_train.csv
│   └── oasis3_multimodal_test.csv
│
└── reports/                       <-- Generated trajectory fan chart plots (.png)
```
## Latest Updates (2026‑10‑19)

- **Quantum circuit upgrade**: Replaced the manual rotation/CNOT layers with a Pennylane `IQPEmbedding` (ZZ‑FeatureMap) using `n_repeats=NUM_LAYERS`. This provides richer pairwise entanglement.
- **Training results**: After the upgrade, training ROC‑AUC reached **0.9805** and the hold‑out evaluation achieved **ROC‑AUC 0.8076**, **Accuracy 0.8258**, **Recall 0.6087**, **F1 0.5926**.
- **Model artifact**: Saved to `saved_models/` and can be re‑used without retraining (load with `UnifiedMultiModalQuantumModel.load()`).
- **Simulation**: Direction #4 sigmoid biological cascade runs for example subjects, showing personalized risk drivers and 10‑year conversion probabilities.

These changes are reflected in the code base and improve model interpretability and performance.
