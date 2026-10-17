"""
Unified Quantum-Embedded Machine Learning Model for Dementia Risk.
Fuses PennyLane Quantum Feature Layer with Gradient-Boosted Trees into a Single Model,
with Per-Prediction Biomarker Identification using TreeSHAP.
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
import pennylane as qml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

NUM_QUBITS = 4
NUM_LAYERS = 2

dev = qml.device("default.qubit", wires=NUM_QUBITS)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_feature_circuit(features, weights):
    """
    Parametrized Quantum Circuit extracting 4 non-linear expectation values.
    features: (NUM_QUBITS,)
    weights: (NUM_LAYERS, NUM_QUBITS, 3)
    """
    qml.AngleEmbedding(features, wires=range(NUM_QUBITS), rotation="Y")

    for layer in range(weights.shape[0]):
        for q in range(NUM_QUBITS):
            qml.Rot(weights[layer, q, 0], weights[layer, q, 1], weights[layer, q, 2], wires=q)
        for q in range(NUM_QUBITS):
            qml.CNOT(wires=[q, (q + 1) % NUM_QUBITS])

    # Measure expectation values across all qubits to produce a 4D quantum representation
    return [qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)]


class UnifiedQuantumXGBoost:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=NUM_QUBITS, random_state=random_state)
        self.pca_scaling_factor = 1.0
        self.xgb_model = None

        # Fixed, reproducible quantum variational parameters for the Hilbert projection
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        self.quantum_weights = torch.tensor(
            np.random.uniform(-np.pi, np.pi, (NUM_LAYERS, NUM_QUBITS, 3)),
            dtype=torch.float32
        )

        self.classical_features = [
            "Age",
            "Normalized_Hippo",
            "Normalized_Cortex",
            "Normalized_SubCortGray",
            "Normalized_WhiteMatter",
            "Hippo_Asymmetry"
        ]
        self.quantum_features = [
            "Quantum_Entangled_Q0",
            "Quantum_Entangled_Q1",
            "Quantum_Entangled_Q2",
            "Quantum_Entangled_Q3"
        ]
        self.all_features = self.classical_features + self.quantum_features
        self.normative_stats = {}

    def extract_quantum_features(self, X_classical_scaled):
        """
        Passes PCA-compressed classical features through the PennyLane circuit
        to produce non-linear quantum interaction features.
        """
        X_pca = self.pca.transform(X_classical_scaled) * self.pca_scaling_factor
        X_tensor = torch.tensor(X_pca, dtype=torch.float32)

        quantum_feats = []
        for i in range(len(X_tensor)):
            q_out = quantum_feature_circuit(X_tensor[i], self.quantum_weights)
            # q_out is a list of 4 tensors
            q_vals = [float(val.detach().item()) for val in q_out]
            quantum_feats.append(q_vals)

        return np.array(quantum_feats)

    def fit(self, train_df):
        print("\n" + "=" * 70)
        print("TRAINING UNIFIED QUANTUM-EMBEDDED XGBOOST MODEL")
        print("=" * 70)

        # 1. Classical Feature Extraction & Scaling
        X_raw = train_df[self.classical_features].values
        y = train_df["target"].values

        # Store normative healthy reference stats for explainability
        healthy_subset = train_df[train_df["target"] == 0]
        for col in self.classical_features:
            self.normative_stats[col] = {
                "mean": float(healthy_subset[col].mean()),
                "std": float(healthy_subset[col].std()),
                "p25": float(healthy_subset[col].quantile(0.25)),
                "p75": float(healthy_subset[col].quantile(0.75))
            }

        X_scaled = self.scaler.fit_transform(X_raw)
        X_pca_raw = self.pca.fit_transform(X_scaled)
        pca_max = np.max(np.abs(X_pca_raw))
        self.pca_scaling_factor = float(np.pi / (pca_max + 1e-6))

        # 2. Extract Quantum Features via PennyLane
        print("[1/2] Computing PennyLane Quantum Hilbert Space Embeddings...")
        X_quantum = self.extract_quantum_features(X_scaled)
        print(f"      Generated {X_quantum.shape[1]} Quantum Entanglement Features.")

        # 3. Form Unified Feature Matrix (Classical + Quantum)
        X_unified = np.hstack([X_raw, X_quantum])
        print(f"      Unified Feature Representation Shape: {X_unified.shape}")

        # 4. Train Single Unified XGBoost Classifier
        print("[2/2] Training Unified Gradient-Boosted Quantum-Classical Model...")
        pos_weight = (len(y) - sum(y)) / sum(y)
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=140,
            max_depth=4,
            learning_rate=0.035,
            scale_pos_weight=pos_weight,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=self.random_state,
            eval_metric="logloss"
        )
        self.xgb_model.fit(X_unified, y)
        train_probs = self.xgb_model.predict_proba(X_unified)[:, 1]
        print(f"Unified Model Training ROC-AUC: {roc_auc_score(y, train_probs):.4f}")
        print("Training Complete.")

    def transform_to_unified(self, X_classical_raw):
        """
        Transforms raw classical features into the unified 10D feature matrix.
        """
        X_scaled = self.scaler.transform(X_classical_raw)
        X_quantum = self.extract_quantum_features(X_scaled)
        return np.hstack([X_classical_raw, X_quantum])

    def predict_risk(self, X_classical_raw):
        """
        Returns the single unified dementia risk probability in [0, 1].
        """
        X_unified = self.transform_to_unified(X_classical_raw)
        return self.xgb_model.predict_proba(X_unified)[:, 1]

    def explain_prediction(self, patient_features_dict):
        """
        Computes exact TreeSHAP local biomarker contributions for a single patient scan,
        identifying the primary driving biomarker and protective factors.
        """
        # Convert dict or Series to numpy
        raw_vals = [float(patient_features_dict[col]) for col in self.classical_features]
        X_raw = np.array([raw_vals])
        X_unified = self.transform_to_unified(X_raw)

        # Single Unified Prediction
        prob = float(self.xgb_model.predict_proba(X_unified)[:, 1][0])

        # Exact TreeSHAP values via XGBoost booster
        booster = self.xgb_model.get_booster()
        dmat = xgb.DMatrix(X_unified, feature_names=self.all_features)
        contribs = booster.predict(dmat, pred_contribs=True)[0]  # shape: (num_features + 1,)
        shap_values = contribs[:-1]
        base_value = contribs[-1]

        # Convert SHAP log-odds contributions into approximate percentage impact
        # Sigmoid probability shift
        p_base = 1.0 / (1.0 + np.exp(-base_value))
        p_final = 1.0 / (1.0 + np.exp(- (base_value + np.sum(shap_values))))
        total_delta = p_final - p_base

        # Attribution dictionary
        attribution = {}
        for feat_name, shap_val, val in zip(self.all_features, shap_values, X_unified[0]):
            # Proportional contribution to probability shift
            pct_impact = float(shap_val)
            attribution[feat_name] = {
                "shap_value": float(shap_val),
                "measured_value": float(val),
                "is_quantum": feat_name.startswith("Quantum")
            }

        # Rank biomarkers by absolute SHAP impact
        ranked = sorted(attribution.items(), key=lambda x: abs(x[1]["shap_value"]), reverse=True)

        # Identify Top Risk Driver and Top Protective Factor
        drivers = [item for item in ranked if item[1]["shap_value"] > 0]
        protective = [item for item in ranked if item[1]["shap_value"] < 0]

        friendly_names = {
            "Normalized_Hippo": "Hippocampal Volume Atrophy (Normalized to ICV)",
            "Normalized_Cortex": "Cortical Gray Matter Thinning",
            "Normalized_SubCortGray": "Subcortical Gray Matter Loss",
            "Normalized_WhiteMatter": "Cerebral White Matter Loss",
            "Hippo_Asymmetry": "Hemispheric Hippocampal Asymmetry",
            "Age": "Biological Aging",
            "Quantum_Entangled_Q0": "Quantum Entanglement Q0 (Non-linear Atrophy Interaction)",
            "Quantum_Entangled_Q1": "Quantum Entanglement Q1 (Multi-Region Interaction)",
            "Quantum_Entangled_Q2": "Quantum Entanglement Q2 (Geometric Hilbert Projection)",
            "Quantum_Entangled_Q3": "Quantum Entanglement Q3 (Non-linear Volumetric Anomaly)"
        }

        def format_item(item):
            if item is None:
                return None
            k, v = item
            return {
                "feature": k,
                "friendly_name": friendly_names.get(k, k),
                "shap_value": v["shap_value"],
                "measured_value": v["measured_value"],
                "is_risk_factor": v["shap_value"] > 0
            }

        top_risk_driver = format_item(drivers[0]) if len(drivers) > 0 else None
        top_protective = format_item(protective[0]) if len(protective) > 0 else None

        # Overall primary biomarker
        primary_biomarker = top_risk_driver if (prob >= 0.40 and top_risk_driver) else (top_protective or top_risk_driver)

        return {
            "predicted_risk": prob,
            "primary_biomarker": primary_biomarker,
            "primary_risk_driver": top_risk_driver,
            "primary_protective_biomarker": top_protective,
            "risk_drivers": [
                {
                    "feature": k,
                    "friendly_name": friendly_names.get(k, k),
                    "shap_value": v["shap_value"],
                    "measured_value": v["measured_value"]
                }
                for k, v in drivers
            ],
            "protective_factors": [
                {
                    "feature": k,
                    "friendly_name": friendly_names.get(k, k),
                    "shap_value": v["shap_value"],
                    "measured_value": v["measured_value"]
                }
                for k, v in protective
            ]
        }

    def evaluate(self, test_df, name="Holdout Test Set"):
        X_test_raw = test_df[self.classical_features].values
        y_test = test_df["target"].values

        probs = self.predict_risk(X_test_raw)
        preds = (probs >= 0.5).astype(int)

        auc = roc_auc_score(y_test, probs)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)

        print(f"\n================ Evaluation on {name} ================")
        print(f"ROC-AUC:   {auc:.4f}")
        print(f"Accuracy:  {acc:.4f} ({acc:.1%})")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f} ({rec:.1%})")
        print(f"F1-Score:  {f1:.4f}")
        return {"ROC-AUC": auc, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1}

    def save(self, save_dir="saved_models"):
        os.makedirs(save_dir, exist_ok=True)
        # 1. Save Preprocessors & Quantum Configuration
        config = {
            "scaler": self.scaler,
            "pca": self.pca,
            "pca_scaling_factor": self.pca_scaling_factor,
            "quantum_weights": self.quantum_weights,
            "classical_features": self.classical_features,
            "quantum_features": self.quantum_features,
            "all_features": self.all_features,
            "normative_stats": self.normative_stats
        }
        joblib.dump(config, os.path.join(save_dir, "unified_preprocessor.joblib"))

        # 2. Save Unified XGBoost Model
        self.xgb_model.save_model(os.path.join(save_dir, "unified_xgboost_model.json"))
        print(f"Unified model and quantum preprocessors saved to: {save_dir}/")

    @classmethod
    def load(cls, save_dir="saved_models"):
        instance = cls()
        config = joblib.load(os.path.join(save_dir, "unified_preprocessor.joblib"))
        instance.scaler = config["scaler"]
        instance.pca = config["pca"]
        instance.pca_scaling_factor = config["pca_scaling_factor"]
        instance.quantum_weights = config["quantum_weights"]
        instance.classical_features = config["classical_features"]
        instance.quantum_features = config["quantum_features"]
        instance.all_features = config["all_features"]
        instance.normative_stats = config["normative_stats"]

        instance.xgb_model = xgb.XGBClassifier()
        instance.xgb_model.load_model(os.path.join(save_dir, "unified_xgboost_model.json"))
        return instance
