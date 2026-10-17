"""
Hybrid Quantum (PennyLane) + Classical (XGBoost) Ensemble Model for Dementia Risk.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import pennylane as qml

# -------------------------------------------------------------
# 1. PennyLane Quantum Circuit Architecture
# -------------------------------------------------------------
NUM_QUBITS = 4
NUM_LAYERS = 3

dev = qml.device("default.qubit", wires=NUM_QUBITS)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(features, weights):
    """
    Parametrized Quantum Circuit (VQC) with Angle Embedding and Entangling layers.
    features: Tensor of shape (NUM_QUBITS,)
    weights: Tensor of shape (NUM_LAYERS, NUM_QUBITS, 3)
    """
    # Quantum Feature Map (Angle Embedding)
    qml.AngleEmbedding(features, wires=range(NUM_QUBITS), rotation="Y")

    # Variational Layers
    for layer in range(weights.shape[0]):
        for q in range(NUM_QUBITS):
            qml.Rot(weights[layer, q, 0], weights[layer, q, 1], weights[layer, q, 2], wires=q)
        # Circular entanglement
        for q in range(NUM_QUBITS):
            qml.CNOT(wires=[q, (q + 1) % NUM_QUBITS])

    # Expectation value measurement on the first qubit
    return qml.expval(qml.PauliZ(0))


class PennyLaneVQC(nn.Module):
    """
    PyTorch Module wrapper for the PennyLane QNode.
    Outputs calibrated probabilities in [0, 1].
    """
    def __init__(self, num_qubits=NUM_QUBITS, num_layers=NUM_LAYERS):
        super(PennyLaneVQC, self).__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.weights = nn.Parameter(0.1 * torch.randn(num_layers, num_qubits, 3))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x is a batch of PCA-reduced features (batch_size, num_qubits)
        batch_size = x.shape[0]
        expvals = []
        for i in range(batch_size):
            ev = quantum_circuit(x[i], self.weights)
            expvals.append(ev)
        expvals = torch.stack(expvals).reshape(-1, 1)
        # Linear layer + Sigmoid to produce probability
        logits = expvals + self.bias
        probs = torch.sigmoid(logits)
        return probs.squeeze(-1)


# -------------------------------------------------------------
# 2. Hybrid Model Manager
# -------------------------------------------------------------
class HybridQuantumXGBoost:
    def __init__(self, alpha=0.65):
        """
        alpha: Weight given to XGBoost (1 - alpha given to Quantum VQC).
        """
        self.alpha = alpha
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=NUM_QUBITS)
        self.xgb_model = None
        self.vqc_model = PennyLaneVQC(num_qubits=NUM_QUBITS, num_layers=NUM_LAYERS)
        self.feature_cols = [
            "Age",
            "Normalized_Hippo",
            "Normalized_Cortex",
            "Normalized_SubCortGray",
            "Normalized_WhiteMatter",
            "Hippo_Asymmetry"
        ]

    def fit(self, train_df, val_df=None, epochs=25, batch_size=32, lr=0.03):
        print("\n================ Training Hybrid Quantum-XGBoost Model ================")
        X_train_raw = train_df[self.feature_cols].values
        y_train = train_df["target"].values

        # 1. Fit Preprocessors
        X_train_scaled = self.scaler.fit_transform(X_train_raw)
        X_train_pca = self.pca.fit_transform(X_train_scaled)
        # Scale PCA features to [-pi, pi] for angle embedding
        pca_max = np.max(np.abs(X_train_pca))
        self.pca_scaling_factor = float(np.pi / (pca_max + 1e-6))
        X_train_quantum = X_train_pca * self.pca_scaling_factor

        # 2. Train Classical XGBoost Model
        print("\n[1/2] Training Classical XGBoost...")
        pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.04,
            scale_pos_weight=pos_weight,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            eval_metric="logloss"
        )
        self.xgb_model.fit(X_train_raw, y_train)
        train_xgb_probs = self.xgb_model.predict_proba(X_train_raw)[:, 1]
        print(f"XGBoost Train ROC-AUC: {roc_auc_score(y_train, train_xgb_probs):.4f}")

        # 3. Train PennyLane Variational Quantum Classifier (VQC)
        print("\n[2/2] Training PennyLane Quantum VQC with PyTorch...")
        device = torch.device("cpu")
        self.vqc_model.to(device)

        X_t = torch.tensor(X_train_quantum, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32)

        # Weighted Binary Cross Entropy for class imbalance
        weight_tensor = torch.tensor([pos_weight], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)
        optimizer = optim.Adam(self.vqc_model.parameters(), lr=lr)

        num_train = len(y_train)
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            self.vqc_model.train()
            total_loss = 0.0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                probs = self.vqc_model(batch_x)
                # Compute BCE loss via numerical log-odds
                eps = 1e-6
                probs_clamped = torch.clamp(probs, eps, 1.0 - eps)
                logits = torch.log(probs_clamped / (1.0 - probs_clamped))
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch_x.size(0)

            epoch_loss = total_loss / num_train
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:2d}/{epochs:2d} | Quantum Loss: {epoch_loss:.5f}")

        print("Quantum VQC Training Complete.")

    def predict_components(self, X_raw):
        """
        Returns (p_xgb, p_quantum, p_ensemble)
        """
        # XGBoost prediction
        p_xgb = self.xgb_model.predict_proba(X_raw)[:, 1]

        # Quantum prediction
        X_scaled = self.scaler.transform(X_raw)
        X_pca = self.pca.transform(X_scaled) * self.pca_scaling_factor
        X_tensor = torch.tensor(X_pca, dtype=torch.float32)

        self.vqc_model.eval()
        with torch.no_grad():
            p_quantum = self.vqc_model(X_tensor).numpy()

        p_ensemble = self.alpha * p_xgb + (1.0 - self.alpha) * p_quantum
        return p_xgb, p_quantum, p_ensemble

    def predict_risk(self, X_raw):
        _, _, p_ensemble = self.predict_components(X_raw)
        return p_ensemble

    def evaluate(self, test_df, name="Holdout Test Set"):
        X_test = test_df[self.feature_cols].values
        y_test = test_df["target"].values

        p_xgb, p_quantum, p_ensemble = self.predict_components(X_test)

        results = {}
        for model_name, probs in [
            ("Classical XGBoost", p_xgb),
            ("PennyLane Quantum VQC", p_quantum),
            ("Hybrid Ensemble", p_ensemble)
        ]:
            preds = (probs >= 0.5).astype(int)
            results[model_name] = {
                "ROC-AUC": roc_auc_score(y_test, probs),
                "Accuracy": accuracy_score(y_test, preds),
                "Precision": precision_score(y_test, preds, zero_division=0),
                "Recall": recall_score(y_test, preds, zero_division=0),
                "F1-Score": f1_score(y_test, preds, zero_division=0),
            }

        print(f"\n================ Evaluation on {name} ================")
        res_df = pd.DataFrame(results).T
        print(res_df.to_string(float_format=lambda x: f"{x:.4f}"))
        return results

    def save(self, save_dir="saved_models"):
        os.makedirs(save_dir, exist_ok=True)
        # 1. Save Preprocessors
        preprocessors = {
            "scaler": self.scaler,
            "pca": self.pca,
            "pca_scaling_factor": self.pca_scaling_factor,
            "feature_cols": self.feature_cols,
            "alpha": self.alpha
        }
        joblib.dump(preprocessors, os.path.join(save_dir, "preprocessors.joblib"))

        # 2. Save XGBoost
        self.xgb_model.save_model(os.path.join(save_dir, "xgboost_model.json"))

        # 3. Save Quantum VQC
        torch.save(self.vqc_model.state_dict(), os.path.join(save_dir, "quantum_vqc.pt"))

        print(f"\nAll hybrid model artifacts successfully saved to: {save_dir}/")

    @classmethod
    def load(cls, save_dir="saved_models"):
        instance = cls()
        prep = joblib.load(os.path.join(save_dir, "preprocessors.joblib"))
        instance.scaler = prep["scaler"]
        instance.pca = prep["pca"]
        instance.pca_scaling_factor = prep["pca_scaling_factor"]
        instance.feature_cols = prep["feature_cols"]
        instance.alpha = prep["alpha"]

        instance.xgb_model = xgb.XGBClassifier()
        instance.xgb_model.load_model(os.path.join(save_dir, "xgboost_model.json"))

        instance.vqc_model = PennyLaneVQC(num_qubits=NUM_QUBITS, num_layers=NUM_LAYERS)
        instance.vqc_model.load_state_dict(torch.load(os.path.join(save_dir, "quantum_vqc.pt")))
        instance.vqc_model.eval()
        return instance
