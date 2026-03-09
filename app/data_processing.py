from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


SCENE_FLOW_MAP = {
    0: 1.00,   # open-field weather mast
    1: 0.92,   # rooftop / structural interference
    2: 1.10,   # duct / channel acceleration
    3: 0.86,   # UAV body / prop wash disturbed region
}

SCENE_NAME_MAP = {
    0: "open_field",
    1: "rooftop",
    2: "duct",
    3: "uav_mount",
}


def _safe_divide(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return a / np.where(np.abs(b) < eps, np.sign(b) * eps + eps, b)



def load_systematic_error_dataset(
    n_samples: int = 8000,
    n_devices: int = 10,
    random_state: int = 42,
):
    rng = np.random.default_rng(random_state)

    # -----------------------------
    # 1) True environmental state
    # -----------------------------
    true_wind = rng.uniform(-14.0, 14.0, n_samples)
    temperature_true = rng.uniform(-15.0, 45.0, n_samples)
    humidity = rng.uniform(20.0, 95.0, n_samples)
    sun_exposure = rng.uniform(0.0, 1.0, n_samples)
    board_power = rng.uniform(0.2, 1.0, n_samples)

    scene_id = rng.integers(0, 4, n_samples)
    device_id = rng.integers(0, n_devices, n_samples)
    mount_tilt_deg = rng.uniform(-12.0, 12.0, n_samples)
    wind_misalignment_deg = rng.uniform(-18.0, 18.0, n_samples)

    # -----------------------------
    # 2) Device-level engineering variability
    # -----------------------------
    device_length_bias = rng.normal(0.0, 7.5e-4, n_devices)
    device_delay_ns_t1 = rng.normal(120.0, 35.0, n_devices)
    device_delay_ns_t2 = rng.normal(-90.0, 35.0, n_devices)
    device_quant_step_ns = rng.choice(np.array([40.0, 60.0, 80.0, 120.0]), size=n_devices)
    device_temp_probe_bias = rng.normal(0.0, 0.8, n_devices)

    # -----------------------------
    # 3) Physical constants and true propagation path
    # -----------------------------
    L0 = 0.12
    T0 = 20.0

    tilt_rad = np.deg2rad(mount_tilt_deg)
    misalign_rad = np.deg2rad(wind_misalignment_deg)

    # true local airflow differs by deployment scene and tilt
    scene_flow_index = np.array([SCENE_FLOW_MAP[int(s)] for s in scene_id])
    v_local = true_wind * scene_flow_index * np.cos(misalign_rad) * (1.0 - 0.0018 * np.abs(mount_tilt_deg))

    # true acoustic path affected by thermal expansion + mounting deviation + small tilt geometry effect
    L_true = L0 * (
        1.0
        + 5.5e-4 * (temperature_true - T0)
        + 1.2e-5 * (temperature_true - T0) ** 2
        + device_length_bias[device_id]
        + 1.5e-4 * np.abs(mount_tilt_deg)
    )

    # more realistic true speed of sound with humidity contribution and mild nonlinearity
    c_true = (
        331.3
        + 0.606 * temperature_true
        + 0.0124 * humidity
        + 2.5e-4 * (temperature_true - 20.0) ** 2
    )

    # -----------------------------
    # 4) True travel times
    # -----------------------------
    denom_1 = c_true + v_local
    denom_2 = c_true - v_local
    denom_1 = np.clip(denom_1, 250.0, None)
    denom_2 = np.clip(denom_2, 250.0, None)

    t1_true = L_true / denom_1
    t2_true = L_true / denom_2

    # -----------------------------
    # 5) Systematic measurement distortions (weak random noise, strong structured bias)
    # -----------------------------
    # temperature probe representativeness error
    temp_shell_bias = 2.6 * sun_exposure + 1.3 * board_power + 0.25 * np.abs(mount_tilt_deg) / 12.0
    temperature_measured = (
        temperature_true
        + temp_shell_bias
        + device_temp_probe_bias[device_id]
        + rng.normal(0.0, 0.08, n_samples)
    )

    # baseline model uses simplified temperature compensation (no humidity, no quadratic term)
    c_model = 331.3 + 0.606 * temperature_measured

    # channel asymmetry / unequal delays in ns
    asym_t1 = (device_delay_ns_t1[device_id] + 22.0 * sun_exposure + 1.8 * mount_tilt_deg) * 1e-9
    asym_t2 = (device_delay_ns_t2[device_id] - 18.0 * sun_exposure - 1.5 * mount_tilt_deg) * 1e-9

    # wind-speed-dependent nonlinear timing bias
    nonlinear_timing = (14.0e-9 * np.abs(true_wind) + 5.0e-9 * (true_wind ** 2) / 100.0)

    t1_meas = t1_true + asym_t1 + nonlinear_timing
    t2_meas = t2_true + asym_t2 + nonlinear_timing * 0.82

    # low-cost hardware time quantization
    quant_step_ns = device_quant_step_ns[device_id]
    quant_step_s = quant_step_ns * 1e-9
    t1_meas = np.round(t1_meas / quant_step_s) * quant_step_s
    t2_meas = np.round(t2_meas / quant_step_s) * quant_step_s

    # retain a very weak random jitter so plots are not unnaturally perfect
    t1_meas = t1_meas + rng.normal(0.0, 2.0e-9, n_samples)
    t2_meas = t2_meas + rng.normal(0.0, 2.0e-9, n_samples)

    # -----------------------------
    # 6) Baseline inversion with intentionally simplified assumptions
    # -----------------------------
    measured_wind = 0.5 * L0 * (1.0 / t1_meas - 1.0 / t2_meas)

    # -----------------------------
    # 7) Feature engineering for RF
    # -----------------------------
    dt = t2_meas - t1_meas
    ts = t2_meas + t1_meas
    abs_measured_wind = np.abs(measured_wind)
    measured_wind_sq = measured_wind ** 2
    temp_x_speed = temperature_measured * measured_wind
    asymmetry_ratio = _safe_divide(dt, ts)
    temp_shell_bias_proxy = 2.6 * sun_exposure + 1.3 * board_power

    df = pd.DataFrame(
        {
            "true_wind": true_wind,
            "temperature_true": temperature_true,
            "temperature_measured": temperature_measured,
            "humidity": humidity,
            "t1": t1_meas,
            "t2": t2_meas,
            "measured_wind": measured_wind,
            "dt": dt,
            "ts": ts,
            "abs_measured_wind": abs_measured_wind,
            "measured_wind_sq": measured_wind_sq,
            "temp_x_speed": temp_x_speed,
            "mount_tilt_deg": mount_tilt_deg,
            "wind_misalignment_deg": wind_misalignment_deg,
            "sun_exposure": sun_exposure,
            "board_power": board_power,
            "scene_id": scene_id.astype(float),
            "device_id": device_id.astype(float),
            "quant_step_ns": quant_step_ns,
            "asymmetry_ratio": asymmetry_ratio,
            "scene_flow_index": scene_flow_index,
            "temp_shell_bias_proxy": temp_shell_bias_proxy,
        }
    )

    feature_columns = [
        "temperature_measured",
        "humidity",
        "t1",
        "t2",
        "measured_wind",
        "dt",
        "ts",
        "abs_measured_wind",
        "measured_wind_sq",
        "temp_x_speed",
        "mount_tilt_deg",
        "wind_misalignment_deg",
        "sun_exposure",
        "board_power",
        "scene_id",
        "device_id",
        "quant_step_ns",
        "asymmetry_ratio",
        "scene_flow_index",
        "temp_shell_bias_proxy",
    ]

    meta = {
        "feature_columns": feature_columns,
        "scene_name_map": SCENE_NAME_MAP,
    }
    return df, meta



def train_test_split_dataset(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    baseline_col: str,
    test_size: float = 0.25,
    random_state: int = 42,
):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

    X_train = train_df[feature_cols].reset_index(drop=True)
    X_test = test_df[feature_cols].reset_index(drop=True)
    y_train = train_df[target_col].to_numpy()
    y_test = test_df[target_col].to_numpy()
    base_train = train_df[baseline_col].to_numpy()
    base_test = test_df[baseline_col].to_numpy()

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        base_train,
        base_test,
        train_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )
