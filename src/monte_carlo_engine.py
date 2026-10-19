"""
Monte Carlo Longitudinal Biological Cascade Trajectory Engine (Direction #4).
Implements a non-linear sigmoidal neurodegenerative cascade (Jack et al. model):
- Cognitive reserve delays clinical symptom onset (buffering phase).
- Vascular burden (stroke, hypertension) accelerates atrophy inflection.
- Projects 10-year forward trajectories with 95% Confidence Intervals.
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class DynamicCascadeSimulator:
    def __init__(self, model, threshold=0.50, num_simulations=1000, horizon_years=10, random_seed=42):
        self.model = model
        self.threshold = threshold
        self.num_simulations = num_simulations
        self.horizon_years = horizon_years
        self.random_seed = random_seed

    def simulate_patient(self, patient_row):
        np.random.seed(self.random_seed)

        # Baseline clinical & neuroimaging metrics
        age_0 = float(patient_row["Age"])
        hippo_0 = float(patient_row["TOTAL_HIPPOCAMPUS_VOLUME"])
        icv_0 = float(patient_row["IntraCranialVol"])

        # Automatically calculate normalized metrics if raw volumes provided
        if "Normalized_Cortex" in patient_row:
            cortex_norm_0 = float(patient_row["Normalized_Cortex"])
        else:
            cortex_norm_0 = float(patient_row.get("CortexVol", 420000.0)) / icv_0

        if "Normalized_SubCortGray" in patient_row:
            subcort_norm_0 = float(patient_row["Normalized_SubCortGray"])
        else:
            subcort_norm_0 = float(patient_row.get("SubCortGrayVol", 52000.0)) / icv_0

        if "Normalized_WhiteMatter" in patient_row:
            wm_norm_0 = float(patient_row["Normalized_WhiteMatter"])
        else:
            wm_norm_0 = float(patient_row.get("CorticalWhiteMatterVol", 420000.0)) / icv_0

        asym_0 = float(patient_row.get("Hippo_Asymmetry", 0.05))

        # Multi-modal modifiers
        cog_reserve = float(patient_row.get("Cognitive_Reserve_Buffer", 0.95))
        vasc_burden = float(patient_row.get("Vascular_Burden_Index", 0.0))
        fam_history = float(patient_row.get("Family_Dementia_History", 0.0))
        norm_hippo_mean = 0.00508
        norm_hippo_std = 0.00069
        hippo_z_0 = (hippo_0 / icv_0 - norm_hippo_mean) / norm_hippo_std
        masked_index_0 = float(np.maximum(0.0, -hippo_z_0) * cog_reserve)

        # Build baseline feature dict for model explanation
        baseline_dict = {
            "Age": age_0,
            "Normalized_Hippo": hippo_0 / icv_0,
            "Normalized_Cortex": cortex_norm_0,
            "Normalized_SubCortGray": subcort_norm_0,
            "Normalized_WhiteMatter": wm_norm_0,
            "Hippo_Asymmetry": asym_0,
            "Reserve_Masked_Index": masked_index_0,
            "Cognitive_Reserve_Buffer": cog_reserve,
            "Vascular_Burden_Index": vasc_burden,
            "Family_Dementia_History": fam_history
        }

        explanation = self.model.explain_prediction(baseline_dict)
        p_base = float(explanation["predicted_risk"])

        # -------------------------------------------------------------
        # Direction #4: Personalized Sigmoidal Biological Cascade
        # Jack et al. non-linear neurodegenerative model:
        # Base annual atrophy is accelerated by vascular burden (+0.5% per index point)
        # and family history (+0.4%), while buffered by cognitive reserve.
        # -------------------------------------------------------------
        base_rate = p_base * 0.026 + (1.0 - p_base) * 0.009
        vasc_acceleration = 1.0 + (vasc_burden * 0.18) + (fam_history * 0.10)
        mu_atrophy = base_rate * vasc_acceleration
        sigma_atrophy = 0.006

        years = np.arange(0, self.horizon_years + 1)
        num_years = len(years)

        risk_trajectories = np.zeros((self.num_simulations, num_years))
        risk_trajectories[:, 0] = p_base

        sim_hippo = np.full(self.num_simulations, hippo_0)
        sim_cortex = np.full(self.num_simulations, cortex_norm_0)
        sim_reserve = np.full(self.num_simulations, cog_reserve)

        crossing_times = [None] * self.num_simulations
        if p_base >= self.threshold:
            for s in range(self.num_simulations):
                crossing_times[s] = 0.0

        for y_idx in range(1, num_years):
            year = years[y_idx]
            sim_age = age_0 + year

            # Non-linear Sigmoidal Progression:
            # Acceleration steepens in the middle phase (inflection point around Year 3-6)
            sigmoidal_factor = 1.0 + 0.40 / (1.0 + np.exp(-(year - 4.5) / 1.5))

            annual_losses = np.random.normal(mu_atrophy * sigmoidal_factor, sigma_atrophy, size=self.num_simulations)
            annual_losses = np.clip(annual_losses, 0.001, 0.12)

            sim_hippo *= (1.0 - annual_losses)
            sim_cortex *= (1.0 - annual_losses * 0.45)
            # Cognitive reserve gradually depletes under cumulative atrophy
            sim_reserve *= (1.0 - annual_losses * 0.30)

            sim_norm_hippo = sim_hippo / icv_0
            sim_hippo_z = (sim_norm_hippo - norm_hippo_mean) / norm_hippo_std
            sim_masked = np.maximum(0.0, -sim_hippo_z) * sim_reserve

            batch_features = np.column_stack([
                np.full(self.num_simulations, sim_age),
                sim_norm_hippo,
                sim_cortex,
                np.full(self.num_simulations, subcort_norm_0),
                np.full(self.num_simulations, wm_norm_0),
                np.full(self.num_simulations, asym_0),
                sim_masked,
                sim_reserve,
                np.full(self.num_simulations, vasc_burden),
                np.full(self.num_simulations, fam_history)
            ])

            # Batch predict
            p_t_all = []
            chunk_size = 128
            for i in range(0, self.num_simulations, chunk_size):
                chunk = batch_features[i:i + chunk_size]
                p_chunk = self.model.predict_risk(chunk)
                p_t_all.append(p_chunk)
            p_t = np.concatenate(p_t_all)
            risk_trajectories[:, y_idx] = p_t

            for s in range(self.num_simulations):
                if crossing_times[s] is None and p_t[s] >= self.threshold:
                    p_prev = risk_trajectories[s, y_idx - 1]
                    frac = (self.threshold - p_prev) / (p_t[s] - p_prev + 1e-9)
                    crossing_times[s] = (year - 1) + max(0.0, min(1.0, frac))

        valid_crossings = [t for t in crossing_times if t is not None]
        conversion_prob_10yr = len(valid_crossings) / self.num_simulations

        if len(valid_crossings) > 0:
            median_time = float(np.median(valid_crossings))
            ci_lower = float(np.percentile(valid_crossings, 2.5))
            ci_upper = float(np.percentile(valid_crossings, 97.5))
        else:
            median_time = None
            ci_lower = None
            ci_upper = None

        return {
            "p_base": p_base,
            "explanation": explanation,
            "mu_atrophy_annual": float(mu_atrophy * 100),
            "vasc_acceleration_multiplier": float(vasc_acceleration),
            "median_time_to_diagnosis": median_time,
            "ci_95": (ci_lower, ci_upper),
            "conversion_prob_10yr": conversion_prob_10yr,
            "years": years,
            "percentiles": {
                "p5": np.percentile(risk_trajectories, 5, axis=0),
                "p25": np.percentile(risk_trajectories, 25, axis=0),
                "p50": np.percentile(risk_trajectories, 50, axis=0),
                "p75": np.percentile(risk_trajectories, 75, axis=0),
                "p95": np.percentile(risk_trajectories, 95, axis=0)
            },
            "trajectories_sample": risk_trajectories[:50]
        }

    def plot_trajectory(self, results, patient_id="Holdout Patient", save_path="reports/monte_carlo_trajectory.png"):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        years = results["years"]
        pct = results["percentiles"]

        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

        for path in results["trajectories_sample"]:
            ax.plot(years, path, color="#90caf9", alpha=0.25, linewidth=1)

        ax.fill_between(years, pct["p5"], pct["p95"], color="#1565c0", alpha=0.15, label="90% Prediction Band (P5 - P95)")
        ax.fill_between(years, pct["p25"], pct["p75"], color="#1565c0", alpha=0.30, label="50% Interquartile Range (P25 - P75)")
        ax.plot(years, pct["p50"], color="#0d47a1", linewidth=3, label="Median Sigmoidal Trajectory (P50)")
        ax.axhline(self.threshold, color="#c62828", linestyle="--", linewidth=2, label=f"Clinical Diagnosis Threshold ({self.threshold:.0%})")

        med = results["median_time_to_diagnosis"]
        ci = results["ci_95"]
        vasc_acc = results["vasc_acceleration_multiplier"]

        if med is not None:
            ax.axvline(med, color="#c62828", linestyle=":", alpha=0.7)
            info_box = (
                f"Multi-Modal Risk: {results['p_base']:.1%}\n"
                f"Personalized Atrophy: {results['mu_atrophy_annual']:.2f}%/yr\n"
                f"Vascular Acceleration: {vasc_acc:.2f}x\n"
                f"10-Yr Conversion Prob: {results['conversion_prob_10yr']:.1%}\n"
                f"Diagnosis Window (if applicable):\n"
                f"  Median: {med:.1f} yrs\n"
                f"  95% CI: [{ci[0]:.1f} - {ci[1]:.1f}] yrs"
            )
            ax.text(
                0.03, 0.95, info_box, transform=ax.transAxes, fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#c62828", alpha=0.92)
            )
        else:
            ax.text(
                0.03, 0.95,
                f"Multi-Modal Risk: {results['p_base']:.1%}\nStable Normal (10-Yr Risk < {self.threshold:.0%})\nPreserved Reserve Buffer",
                transform=ax.transAxes, fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#2e7d32", alpha=0.92)
            )

        ax.set_title(f"Personalized Sigmoidal Dementia Trajectory ({patient_id})", fontsize=13, fontweight="bold")
        ax.set_xlabel("Years Forward from Baseline Assessment", fontsize=11)
        ax.set_ylabel("Multi-Modal Dementia Conversion Risk", fontsize=11)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(0, self.horizon_years)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower right", fontsize=9)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return save_path
