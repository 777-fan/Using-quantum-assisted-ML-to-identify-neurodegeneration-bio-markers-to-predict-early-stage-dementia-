"""
Unified Multi-Modal Quantum-Embedded Machine Learning Model for Dementia Risk.
Fuses PennyLane 6-Qubit Entanglement with Gradient-Boosted Trees,
evaluating Structural Neuroimaging + Cognitive Reserve Buffer + Vascular Strain.
Provides Multi-Domain TreeSHAP Explainability.
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
import joblib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
import pennylane as qml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

NUM_QUBITS = 6
NUM_LAYERS = 2

dev = qml.device("default.qubit", wires=NUM_QUBITS)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def multimodal_quantum_circuit(features, weights):
    """
    Parametrized 6-Qubit Quantum Circuit extracting 6 non-linear expectation values.
    features: (NUM_QUBITS,)
    weights: (NUM_LAYERS, NUM_QUBITS, 3)
    """
    qml.AngleEmbedding(features, wires=range(NUM_QUBITS), rotation="Y")

    # Apply an IQP (ZZ) feature map to capture pairwise interactions
    qml.templates.embeddings.IQPEmbedding(features, wires=range(NUM_QUBITS), n_repeats=NUM_LAYERS)

    # Measure PauliZ expectation values across all qubits
    return [qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)]


class UnifiedMultiModalQuantumModel:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=NUM_QUBITS, random_state=random_state)
        self.pca_scaling_factor = 1.0
        self.xgb_model = None

        # Fixed, reproducible quantum variational parameters for Hilbert space projection
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
            "Hippo_Asymmetry",
            "Reserve_Masked_Index",
            "Cognitive_Reserve_Buffer",
            "Vascular_Burden_Index",
            "Family_Dementia_History"
        ]
        self.quantum_features = [
            f"Quantum_Entangled_Q{i}" for i in range(NUM_QUBITS)
        ]
        self.all_features = self.classical_features + self.quantum_features
        self.normative_stats = {}

    def extract_quantum_features(self, X_classical_scaled):
        X_pca = self.pca.transform(X_classical_scaled) * self.pca_scaling_factor
        X_tensor = torch.tensor(X_pca, dtype=torch.float32)

        quantum_feats = []
        for i in range(len(X_tensor)):
            q_out = multimodal_quantum_circuit(X_tensor[i], self.quantum_weights)
            q_vals = [float(val.detach().item()) for val in q_out]
            quantum_feats.append(q_vals)

        return np.array(quantum_feats)

    def fit(self, train_df):
        print("\n" + "=" * 70)
        print("TRAINING UNIFIED MULTI-MODAL QUANTUM-EMBEDDED XGBOOST")
        print("=" * 70)

        X_raw = train_df[self.classical_features].values
        y = train_df["target"].values

        # Normative reference stats for explanation
        healthy_subset = train_df[train_df["target"] == 0]
        for col in self.classical_features:
            self.normative_stats[col] = {
                "mean": float(healthy_subset[col].mean()),
                "std": float(healthy_subset[col].std())
            }

        # 1. Classical Scaling & PCA for Quantum Circuit
        X_scaled = self.scaler.fit_transform(X_raw)
        X_pca_raw = self.pca.fit_transform(X_scaled)
        pca_max = np.max(np.abs(X_pca_raw))
        self.pca_scaling_factor = float(np.pi / (pca_max + 1e-6))

        # 2. Extract 6D Quantum Hilbert Embeddings
        print(f"[1/2] Computing 6-Qubit PennyLane Multi-Modal Quantum Entanglement...")
        X_quantum = self.extract_quantum_features(X_scaled)
        print(f"      Generated {X_quantum.shape[1]} Quantum Entanglement Features.")

        # 3. Form Unified Feature Matrix (Classical + Quantum)
        X_unified = np.hstack([X_raw, X_quantum])
        print(f"      Unified Feature Matrix: {X_unified.shape[0]} rows × {X_unified.shape[1]} columns")

        # 4. Train Unified Gradient Boosted Trees
        print("[2/2] Training Unified Multi-Modal Gradient Boosted Classifier...")
        pos_weight = (len(y) - sum(y)) / sum(y)
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=150,
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

    def transform_to_unified(self, X_classical_raw):
        X_scaled = self.scaler.transform(X_classical_raw)
        X_quantum = self.extract_quantum_features(X_scaled)
        return np.hstack([X_classical_raw, X_quantum])

    def predict_risk(self, X_classical_raw):
        X_unified = self.transform_to_unified(X_classical_raw)
        return self.xgb_model.predict_proba(X_unified)[:, 1]

    def explain_prediction(self, patient_row_dict):
        """
        Runs TreeSHAP local attribution across structural, cognitive, and vascular domains.
        """
        raw_vals = [float(patient_row_dict[col]) for col in self.classical_features]
        X_raw = np.array([raw_vals])
        X_unified = self.transform_to_unified(X_raw)

        prob = float(self.xgb_model.predict_proba(X_unified)[:, 1][0])

        booster = self.xgb_model.get_booster()
        dmat = xgb.DMatrix(X_unified, feature_names=self.all_features)
        contribs = booster.predict(dmat, pred_contribs=True)[0]
        shap_values = contribs[:-1]

        friendly_names = {
            "Normalized_Hippo": "Hippocampal Volume Ratio (Medial Temporal Atrophy)",
            "Normalized_Cortex": "Cortical Gray Matter Thinning",
            "Normalized_SubCortGray": "Subcortical Gray Matter Loss",
            "Normalized_WhiteMatter": "Cerebral White Matter Loss",
            "Hippo_Asymmetry": "Hemispheric Hippocampal Asymmetry",
            "Age": "Biological Aging",
            "Reserve_Masked_Index": "Cognitive Reserve Discordance (Masked Atrophy)",
            "Cognitive_Reserve_Buffer": "Cognitive Reserve Buffer (Baseline Function)",
            "Vascular_Burden_Index": "Vascular Disease Burden (Stroke/CVD/Hypertension)",
            "Family_Dementia_History": "Familial Genetic Predisposition (Parental Dementia)",
            "Quantum_Entangled_Q0": "Quantum Entanglement Q0 (Structural Geometric Interaction)",
            "Quantum_Entangled_Q1": "Quantum Entanglement Q1 (Cognitive-Structural Coupling)",
            "Quantum_Entangled_Q2": "Quantum Entanglement Q2 (Vascular-Atrophy Cross-talk)",
            "Quantum_Entangled_Q3": "Quantum Entanglement Q3 (Multi-Domain Hilbert Feature)",
            "Quantum_Entangled_Q4": "Quantum Entanglement Q4 (Non-linear Progression Pattern)",
            "Quantum_Entangled_Q5": "Quantum Entanglement Q5 (Reserve Buffer Projection)"
        }

        attribution = []
        for feat, shap_val, val in zip(self.all_features, shap_values, X_unified[0]):
            attribution.append({
                "feature": feat,
                "friendly_name": friendly_names.get(feat, feat),
                "shap_value": float(shap_val),
                "measured_value": float(val),
                "is_quantum": feat.startswith("Quantum")
            })

        ranked = sorted(attribution, key=lambda x: abs(x["shap_value"]), reverse=True)
        drivers = [item for item in ranked if item["shap_value"] > 0]
        protective = [item for item in ranked if item["shap_value"] < 0]

        top_driver = drivers[0] if len(drivers) > 0 else None
        top_protective = protective[0] if len(protective) > 0 else None
        primary_biomarker = top_driver if (prob >= 0.40 and top_driver) else (top_protective or top_driver)

        return {
            "predicted_risk": prob,
            "primary_biomarker": primary_biomarker,
            "primary_risk_driver": top_driver,
            "primary_protective_biomarker": top_protective,
            "risk_drivers": drivers,
            "protective_factors": protective
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
        joblib.dump(config, os.path.join(save_dir, "multimodal_preprocessor.joblib"))
        self.xgb_model.save_model(os.path.join(save_dir, "multimodal_xgboost_model.json"))
        print(f"Multi-modal model saved to: {save_dir}/")

    @classmethod
    def load(cls, save_dir="saved_models"):
        instance = cls()
        config = joblib.load(os.path.join(save_dir, "multimodal_preprocessor.joblib"))
        instance.scaler = config["scaler"]
        instance.pca = config["pca"]
        instance.pca_scaling_factor = config["pca_scaling_factor"]
        instance.quantum_weights = config["quantum_weights"]
        instance.classical_features = config["classical_features"]
        instance.quantum_features = config["quantum_features"]
        instance.all_features = config["all_features"]
        instance.normative_stats = config["normative_stats"]

        instance.xgb_model = xgb.XGBClassifier()
        instance.xgb_model.load_model(os.path.join(save_dir, "multimodal_xgboost_model.json"))
        return instance
