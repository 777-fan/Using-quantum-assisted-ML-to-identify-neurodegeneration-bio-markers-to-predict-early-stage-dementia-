"""
Monte Carlo Longitudinal Trajectory Simulation Engine.
Simulates future neurodegenerative atrophy trajectories and projects time window to dementia diagnosis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

class MonteCarloTrajectorySimulator:
    def __init__(self, hybrid_model, threshold=0.50, num_simulations=1000, horizon_years=10, random_seed=42):
        self.model = hybrid_model
        self.threshold = threshold
        self.num_simulations = num_simulations
        self.horizon_years = horizon_years
        self.random_seed = random_seed

    def simulate_patient(self, patient_row):
        """
        Runs Monte Carlo forward simulation for a given patient's brain scan.
        patient_row: pandas Series or dict containing:
          - Age
          - TOTAL_HIPPOCAMPUS_VOLUME
          - IntraCranialVol
          - Normalized_Cortex
          - Normalized_SubCortGray
          - Normalized_WhiteMatter
          - Hippo_Asymmetry
        """
        np.random.seed(self.random_seed)

        # Baseline measurements
        age_0 = float(patient_row["Age"])
        hippo_0 = float(patient_row["TOTAL_HIPPOCAMPUS_VOLUME"])
        icv_0 = float(patient_row["IntraCranialVol"])
        cortex_norm_0 = float(patient_row["Normalized_Cortex"])
        subcort_norm_0 = float(patient_row["Normalized_SubCortGray"])
        wm_norm_0 = float(patient_row["Normalized_WhiteMatter"])
        asym_0 = float(patient_row["Hippo_Asymmetry"])

        # Initial baseline feature array
        norm_hippo_0 = hippo_0 / icv_0
        baseline_features = np.array([[
            age_0, norm_hippo_0, cortex_norm_0, subcort_norm_0, wm_norm_0, asym_0
        ]])

        explanation = None
        if hasattr(self.model, "explain_prediction"):
            explanation = self.model.explain_prediction(patient_row)
            p_base = float(explanation["predicted_risk"])
            p_xgb_0 = p_base
            p_q_0 = p_base
        else:
            p_xgb_arr, p_q_arr, p_base_arr = self.model.predict_components(baseline_features)
            p_base = float(p_base_arr[0])
            p_xgb_0 = float(p_xgb_arr[0])
            p_q_0 = float(p_q_arr[0])

        # Progression parameters based on initial clinical risk state:
        # Accelerated AD neurodegeneration: mean ~2.8% annual hippo loss, std ~0.8%
        # Normal healthy aging: mean ~1.0% annual hippo loss, std ~0.4%
        mu_atrophy = p_base * 0.028 + (1.0 - p_base) * 0.010
        sigma_atrophy = 0.007

        years = np.arange(0, self.horizon_years + 1)
        num_years = len(years)

        # Matrix to store risk probabilities: shape (num_simulations, num_years)
        risk_trajectories = np.zeros((self.num_simulations, num_years))
        risk_trajectories[:, 0] = p_base

        crossing_times = []

        # Vectorized simulation matrix:
        # Array of current hippocampal volumes for all paths: shape (num_simulations,)
        sim_hippo = np.full(self.num_simulations, hippo_0)
        sim_cortex = np.full(self.num_simulations, cortex_norm_0)

        crossing_times = [None] * self.num_simulations
        if p_base >= self.threshold:
            for sim in range(self.num_simulations):
                crossing_times[sim] = 0.0

        # Simulate year by year across all paths simultaneously
        for y_idx in range(1, num_years):
            year = years[y_idx]
            sim_age = age_0 + year

            # Stochastic annual atrophy rates for all paths: shape (num_simulations,)
            annual_losses = np.random.normal(mu_atrophy, sigma_atrophy, size=self.num_simulations)
            annual_losses = np.clip(annual_losses, 0.001, 0.10)

            sim_hippo *= (1.0 - annual_losses)
            sim_cortex *= (1.0 - annual_losses * 0.5)

            sim_norm_hippo = sim_hippo / icv_0

            # Batch features for year t: shape (num_simulations, 6)
            batch_features = np.column_stack([
                np.full(self.num_simulations, sim_age),
                sim_norm_hippo,
                sim_cortex,
                np.full(self.num_simulations, subcort_norm_0),
                np.full(self.num_simulations, wm_norm_0),
                np.full(self.num_simulations, asym_0)
            ])

            # Predict risk for all paths at year t in batches of 128
            p_t_all = []
            chunk_size = 128
            for i in range(0, self.num_simulations, chunk_size):
                chunk = batch_features[i:i + chunk_size]
                p_chunk = self.model.predict_risk(chunk)
                p_t_all.append(p_chunk)
            p_t = np.concatenate(p_t_all)
            risk_trajectories[:, y_idx] = p_t

            # Check threshold crossings
            for sim in range(self.num_simulations):
                if crossing_times[sim] is None and p_t[sim] >= self.threshold:
                    p_prev = risk_trajectories[sim, y_idx - 1]
                    frac = (self.threshold - p_prev) / (p_t[sim] - p_prev + 1e-9)
                    crossing_times[sim] = (year - 1) + max(0.0, min(1.0, frac))

        # Filter valid crossing times
        valid_crossings = [t for t in crossing_times if t is not None]

        # Statistical Aggregations
        p5 = np.percentile(risk_trajectories, 5, axis=0)
        p25 = np.percentile(risk_trajectories, 25, axis=0)
        p50 = np.percentile(risk_trajectories, 50, axis=0)
        p75 = np.percentile(risk_trajectories, 75, axis=0)
        p95 = np.percentile(risk_trajectories, 95, axis=0)

        conversion_prob_10yr = len(valid_crossings) / self.num_simulations

        if len(valid_crossings) > 0:
            median_time = float(np.median(valid_crossings))
            ci_lower = float(np.percentile(valid_crossings, 2.5))
            ci_upper = float(np.percentile(valid_crossings, 97.5))
        else:
            median_time = None
            ci_lower = None
            ci_upper = None

        results = {
            "p_base": p_base,
            "p_xgb_0": p_xgb_0,
            "p_quantum_0": p_q_0,
            "explanation": explanation,
            "median_time_to_diagnosis": median_time,
            "ci_95": (ci_lower, ci_upper),
            "conversion_prob_10yr": conversion_prob_10yr,
            "years": years,
            "percentiles": {
                "p5": p5,
                "p25": p25,
                "p50": p50,
                "p75": p75,
                "p95": p95
            },
            "trajectories_sample": risk_trajectories[:50]  # first 50 sample paths
        }

        return results

    def plot_trajectory(self, results, patient_id="Holdout Patient", save_path="reports/monte_carlo_trajectory.png"):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        years = results["years"]
        pct = results["percentiles"]

        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

        # Plot faint individual sample paths
        for path in results["trajectories_sample"]:
            ax.plot(years, path, color="#90caf9", alpha=0.25, linewidth=1)

        # Plot confidence intervals
        ax.fill_between(years, pct["p5"], pct["p95"], color="#1976d2", alpha=0.15, label="90% Prediction Interval (P5 - P95)")
        ax.fill_between(years, pct["p25"], pct["p75"], color="#1976d2", alpha=0.30, label="50% Interquartile Range (P25 - P75)")

        # Plot median trajectory line
        ax.plot(years, pct["p50"], color="#0d47a1", linewidth=3, label="Median Projected Trajectory (P50)")

        # Plot diagnostic threshold
        ax.axhline(self.threshold, color="#d32f2f", linestyle="--", linewidth=2, label=f"Clinical Diagnosis Threshold ({self.threshold:.0%})")

        # Threshold crossing annotation
        if results["median_time_to_diagnosis"] is not None:
            t_med = results["median_time_to_diagnosis"]
            ci = results["ci_95"]
            ax.axvline(t_med, color="#d32f2f", linestyle=":", alpha=0.7)
            annotation_text = (
                f"Baseline Risk: {results['p_base']:.1%}\n"
                f"10-Yr Conversion Prob: {results['conversion_prob_10yr']:.1%}\n"
                f"Estimated Diagnosis Window:\n"
                f"  Median: {t_med:.1f} yrs\n"
                f"  95% CI: [{ci[0]:.1f} - {ci[1]:.1f}] yrs"
            )
            ax.text(
                0.03, 0.95, annotation_text,
                transform=ax.transAxes,
                fontsize=11,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#d32f2f", alpha=0.9)
            )
        else:
            ax.text(
                0.03, 0.95,
                f"Baseline Risk: {results['p_base']:.1%}\nStable Normal (10-Yr Risk < {self.threshold:.0%})",
                transform=ax.transAxes,
                fontsize=11,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#388e3c", alpha=0.9)
            )

        ax.set_title(f"Monte Carlo Dementia Risk Trajectory ({patient_id})", fontsize=14, fontweight="bold")
        ax.set_xlabel("Years Forward from Baseline MRI Scan", fontsize=12)
        ax.set_ylabel("Predicted Alzheimer's Spectrum Risk", fontsize=12)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(0, self.horizon_years)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower right", fontsize=10)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Saved Monte Carlo trajectory plot to: {save_path}")
        return save_path
