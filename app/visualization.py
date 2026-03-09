from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np



def plot_prediction_scatter(y_true, y_baseline, y_corrected, save_path: str):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_baseline, s=12, alpha=0.35, label="Baseline")
    plt.scatter(y_true, y_corrected, s=12, alpha=0.35, label="Corrected")
    lo = min(np.min(y_true), np.min(y_baseline), np.min(y_corrected))
    hi = max(np.max(y_true), np.max(y_baseline), np.max(y_corrected))
    plt.plot([lo, hi], [lo, hi], "k--", linewidth=1.2)
    plt.xlabel("True Wind (m/s)")
    plt.ylabel("Predicted / Measured Wind (m/s)")
    plt.title("Before vs After Correction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()



def plot_error_distribution(err_baseline, err_corrected, save_path: str):
    plt.figure(figsize=(8, 6))
    plt.hist(err_baseline, bins=60, alpha=0.55, density=True, label="Baseline error")
    plt.hist(err_corrected, bins=60, alpha=0.55, density=True, label="Corrected error")
    plt.xlabel("Prediction Error (m/s)")
    plt.ylabel("Density")
    plt.title("Error Distribution Overlay")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()



def plot_error_boxplot(err_baseline, err_corrected, save_path: str):
    plt.figure(figsize=(7, 6))
    plt.boxplot([err_baseline, err_corrected], labels=["Baseline", "Corrected"], showfliers=False)
    plt.ylabel("Prediction Error (m/s)")
    plt.title("Error Boxplot")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()



def plot_abs_error_distribution(err_baseline, err_corrected, save_path: str):
    plt.figure(figsize=(8, 6))
    plt.hist(np.abs(err_baseline), bins=60, alpha=0.55, density=True, label="|Baseline error|")
    plt.hist(np.abs(err_corrected), bins=60, alpha=0.55, density=True, label="|Corrected error|")
    plt.xlabel("Absolute Error (m/s)")
    plt.ylabel("Density")
    plt.title("Absolute Error Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()



def plot_error_vs_feature(feature, err_baseline, err_corrected, xlabel: str, save_path: str):
    plt.figure(figsize=(8, 6))
    plt.scatter(feature, err_baseline, s=10, alpha=0.28, label="Baseline")
    plt.scatter(feature, err_corrected, s=10, alpha=0.28, label="Corrected")
    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.xlabel(xlabel)
    plt.ylabel("Prediction Error (m/s)")
    plt.title(f"Error vs {xlabel}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()



def plot_feature_importance(feature_names, importances, save_path: str, top_k: int = 15):
    feature_names = np.asarray(feature_names)
    importances = np.asarray(importances)
    order = np.argsort(importances)[::-1][:top_k]

    plt.figure(figsize=(10, 6))
    plt.barh(feature_names[order][::-1], importances[order][::-1])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()
