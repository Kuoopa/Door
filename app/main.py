import os
import numpy as np

from data_processing import load_systematic_error_dataset, train_test_split_dataset
from model import ResidualWindCorrector, evaluate_predictions
from visualization import (
    plot_prediction_scatter,
    plot_error_distribution,
    plot_error_boxplot,
    plot_abs_error_distribution,
    plot_error_vs_feature,
    plot_feature_importance,
)


def print_metric_block(title: str, metrics: dict):
    print(f"\n{title}")
    print("-" * len(title))
    print(f"MAE  : {metrics['mae']:.4f}")
    print(f"RMSE : {metrics['rmse']:.4f}")
    print(f"R^2  : {metrics['r2']:.4f}")


def main():
    np.random.seed(42)
    os.makedirs("outputs", exist_ok=True)

    print("Loading systematic-error-dominant ultrasonic wind dataset...")
    df, meta = load_systematic_error_dataset(
        n_samples=9000,
        n_devices=12,
        random_state=42,
    )

    feature_cols = meta["feature_columns"]
    target_col = "true_wind"
    baseline_col = "measured_wind"

    X_train, X_test, y_train, y_test, base_train, base_test, train_df, test_df = train_test_split_dataset(
        df,
        feature_cols=feature_cols,
        target_col=target_col,
        baseline_col=baseline_col,
        test_size=0.25,
        random_state=42,
    )

    print("Training residual random forest corrector...")
    model = ResidualWindCorrector(random_state=42)
    model.fit(X_train, y_train, base_train)
    y_pred = model.predict(X_test, base_test)

    baseline_metrics = evaluate_predictions(y_test, base_test)
    corrected_metrics = evaluate_predictions(y_test, y_pred)

    print_metric_block("Baseline: measured_wind vs true_wind", baseline_metrics)
    print_metric_block("Corrected: residual RF output vs true_wind", corrected_metrics)

    mae_gain = (baseline_metrics["mae"] - corrected_metrics["mae"]) / baseline_metrics["mae"] * 100.0
    rmse_gain = (baseline_metrics["rmse"] - corrected_metrics["rmse"]) / baseline_metrics["rmse"] * 100.0
    print(f"\nMAE improvement : {mae_gain:.2f}%")
    print(f"RMSE improvement: {rmse_gain:.2f}%")

    err_base = base_test - y_test
    err_corr = y_pred - y_test

    plot_prediction_scatter(
        y_true=y_test,
        y_baseline=base_test,
        y_corrected=y_pred,
        save_path="outputs/01_prediction_scatter.png",
    )
    plot_error_distribution(
        err_baseline=err_base,
        err_corrected=err_corr,
        save_path="outputs/02_error_distribution.png",
    )
    plot_error_boxplot(
        err_baseline=err_base,
        err_corrected=err_corr,
        save_path="outputs/03_error_boxplot.png",
    )
    plot_abs_error_distribution(
        err_baseline=err_base,
        err_corrected=err_corr,
        save_path="outputs/04_abs_error_distribution.png",
    )
    plot_error_vs_feature(
        feature=test_df["temperature_true"].values,
        err_baseline=err_base,
        err_corrected=err_corr,
        xlabel="True Temperature (°C)",
        save_path="outputs/05_error_vs_temperature.png",
    )
    plot_error_vs_feature(
        feature=test_df["measured_wind"].values,
        err_baseline=err_base,
        err_corrected=err_corr,
        xlabel="Measured Wind (m/s)",
        save_path="outputs/06_error_vs_measured_wind.png",
    )
    plot_error_vs_feature(
        feature=test_df["mount_tilt_deg"].values,
        err_baseline=err_base,
        err_corrected=err_corr,
        xlabel="Mount Tilt (deg)",
        save_path="outputs/07_error_vs_mount_tilt.png",
    )
    plot_feature_importance(
        feature_names=feature_cols,
        importances=model.feature_importances_,
        save_path="outputs/08_feature_importance.png",
    )

    print("\nFigures saved in ./outputs")
    print("Done.")


if __name__ == "__main__":
    main()
