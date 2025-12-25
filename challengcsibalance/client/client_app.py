"""quickstart-xgboost: Flower / XGBoost ClientApp."""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common.config import unflatten_dict


from challengcsibalance.lib.task import (
    load_data,
    clean_ts,
    convert_timeseries_columns,
    replace_keys
)

# Flower ClientApp


app = ClientApp()


@app.train()
def train(msg: Message, context: Context) -> Message:

    # context.state["actual_activated_clients"] += 1
    print(f"ClientApp {context.node_id} - Starting training round {msg.content['config']['server-round']}")
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    groups_dfs = load_data(msg.content['config']['server-round'], num_partitions)
    groups_dfs = preprocess_groups(groups_dfs)
    X_train, _, y_train, _ = build_train_validation(groups_dfs, train_fraction=0.8)

    # Data audit + outlier removal (IQR-based) on training features
    X_train_df = X_train.copy()
    feature_cols = [col for col in X_train_df.columns if col != "label" and 'time_series' not in col]

    X_train_df, _ = audit_and_remove_outliers(X_train_df, feature_cols, iqr_factor=1.5)
    # Align labels after filtering
    y_train = y_train.iloc[: len(X_train_df)]

    # Min-Max scale training features
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_df)


    model = XGBRegressor(
        n_estimators=50, max_depth=6, learning_rate=0.1,
        subsample=0.4, colsample_bytree=0.4,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1 , num_round=5 ,eval_metric='mae'
    )


    global_round = msg.content["config"]["server-round"]
    if global_round > 1:
        global_model = bytearray(msg.content["arrays"]["0"].numpy().tobytes())
        model.load_model(global_model)

    model.fit(X_train, y_train)

    booster = model.get_booster()
    local_model = booster.save_raw("json")
    model_np = np.frombuffer(local_model, dtype=np.uint8)

    content = RecordDict({
        "arrays": ArrayRecord([model_np]),
        "metrics": MetricRecord({"num-examples": len(X_train)})
    })
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    groups_dfs = load_data(partition_id, num_partitions)
    groups_dfs = preprocess_groups(groups_dfs)
    _, X_val, _, y_val = build_train_validation(groups_dfs, train_fraction=0.8)

    # Data audit + outlier removal (IQR-based) on validation features
    X_val_df = X_val.copy()

    feature_cols = [col for col in X_val_df.columns if col != "label" and 'time_series' not in col]

    X_val_df, _ = audit_and_remove_outliers(X_val_df, feature_cols, iqr_factor=1.5)
    y_val = y_val.iloc[: len(X_val_df)]

    # Min-Max scale validation features
    scaler = MinMaxScaler()
    X_val = scaler.fit_transform(X_val_df)

    bst = XGBRegressor()
    global_model = bytearray(msg.content["arrays"]["0"].numpy().tobytes())
    bst.load_model(global_model)

    metrics = {
        "auc": bst.score(X_val, y_val),
        "num-examples": len(X_val),
        "mae": mean_absolute_error(y_val, bst.predict(X_val))
    }

    return Message(content=RecordDict({"metrics": MetricRecord(metrics)}), reply_to=msg)



def preprocess_groups(groups_dfs):
    """Convert timeseries columns and clean data."""
    for group, users in groups_dfs.items():
        for user_id, df in users.items():
            df = convert_timeseries_columns(df)
            for col in df.columns:
                if col.startswith("timeseries_"):
                    df[col] = df[col].apply(clean_ts)
            df = add_sleep_features(df)
            groups_dfs[group][user_id] = df
    return groups_dfs

def ts_features(ts):
    if ts is None:
        return [np.nan] * 7

    ts = pd.Series(ts).dropna()
    if len(ts) < 2:
        return [np.nan] * 7

    x = np.arange(len(ts))

    slope = np.polyfit(x, ts, 1)[0]

    p25, p50, p75 = np.percentile(ts, [25, 50, 75])
    iqr = p75 - p25

    rmssd = np.sqrt(np.mean(np.diff(ts) ** 2))

    mean = ts.mean()
    std = ts.std()

    return slope, p25, p50, p75, iqr, rmssd, std
def add_sleep_features(df):
    df = df.copy()

    # Sleep composition
    df['deep_sleep_pct'] = np.nan
    df['rem_sleep_pct'] = np.nan
    df['light_sleep_pct'] = np.nan
    df['awake_sleep_pct'] = np.nan
    df['sleep_efficiency'] = np.nan
    df['sleep_day_ptc'] = np.nan

    # HR features
    df['hr_slope'] = np.nan
    df['hr_p25'] = np.nan
    df['hr_p50'] = np.nan
    df['hr_p75'] = np.nan
    df['hr_iqr'] = np.nan
    df['hr_rmssd'] = np.nan
    df['std_hearthrate'] = np.nan

    # Stress features
    df['stress_slope'] = np.nan
    df['stress_p25'] = np.nan
    df['stress_p50'] = np.nan
    df['stress_p75'] = np.nan
    df['stress_iqr'] = np.nan
    df['stress_rmssd'] = np.nan
    df['std_stress'] = np.nan

    # Resp features
    df['resp_slope'] = np.nan
    df['resp_p25'] = np.nan
    df['resp_p50'] = np.nan
    df['resp_p75'] = np.nan
    df['resp_iqr'] = np.nan
    df['resp_rmssd'] = np.nan
    df['std_resp'] = np.nan

    for idx, row in df.iterrows():
        sleep_time = row.get("sleep_sleepTimeSeconds", np.nan)
        if pd.isna(sleep_time) or sleep_time == 0:
            continue

        # ---- Sleep composition
        deep = row.get("sleep_deepSleepSeconds", 0) / sleep_time
        rem = row.get("sleep_remSleepSeconds", 0) / sleep_time
        light = row.get("sleep_lightSleepSeconds", 0) / sleep_time
        awake = row.get("sleep_awakeSleepSeconds", 0) / sleep_time
        efficiency = sleep_time / (sleep_time + row.get("sleep_awakeSleepSeconds", 0))

        df.at[idx, 'deep_sleep_pct'] = deep
        df.at[idx, 'rem_sleep_pct'] = rem
        df.at[idx, 'light_sleep_pct'] = light
        df.at[idx, 'awake_sleep_pct'] = awake
        df.at[idx, 'sleep_efficiency'] = efficiency
        df.at[idx, 'sleep_day_ptc'] = sleep_time / 86400.0

        # ---- HR
        hr_feats = ts_features(row.get("hr_time_series"))
        (
            df.at[idx, 'hr_slope'],
            df.at[idx, 'hr_p25'],
            df.at[idx, 'hr_p50'],
            df.at[idx, 'hr_p75'],
            df.at[idx, 'hr_iqr'],
            df.at[idx, 'hr_rmssd'],
            df.at[idx, 'std_hearthrate'],
        ) = hr_feats

        # ---- Stress
        stress_feats = ts_features(row.get("stress_time_series"))
        (
            df.at[idx, 'stress_slope'],
            df.at[idx, 'stress_p25'],
            df.at[idx, 'stress_p50'],
            df.at[idx, 'stress_p75'],
            df.at[idx, 'stress_iqr'],
            df.at[idx, 'stress_rmssd'],
            df.at[idx, 'std_stress'],
        ) = stress_feats

        # ---- Resp
        resp_feats = ts_features(row.get("resp_time_series"))
        (
            df.at[idx, 'resp_slope'],
            df.at[idx, 'resp_p25'],
            df.at[idx, 'resp_p50'],
            df.at[idx, 'resp_p75'],
            df.at[idx, 'resp_iqr'],
            df.at[idx, 'resp_rmssd'],
            df.at[idx, 'std_resp'],
        ) = resp_feats

    return df



def build_train_validation(groups_dfs, train_fraction=1.0):
    """Aggregate data from all users into single training and validation sets."""
    X_list, y_list = [], []

    for users in groups_dfs.values():
        for df in users.values():
            df = df.dropna(subset=["label"])
            #take everything except label and column that contain 'time_series' in their name
            feature_cols = [col for col in df.columns if col != "label" and 'time_series' not in col]
            X_list.append(df[feature_cols])

            # X_list.append(df[FEATURES])
            y_list.append(df["label"])

    X_all = pd.concat(X_list, axis=0).replace([np.inf, -np.inf], np.nan)
    y_all = pd.concat(y_list, axis=0)

    split_idx = int(len(X_all) * train_fraction)
    X_train, X_val = X_all.iloc[:split_idx], X_all.iloc[split_idx:]
    y_train, y_val = y_all.iloc[:split_idx], y_all.iloc[split_idx:]

    return X_train, X_val, y_train, y_val


def audit_and_remove_outliers(df: pd.DataFrame, feature_cols, iqr_factor: float = 1.5):
    """Audit and clean very low-quality data.
    Steps:
      - Numeric coercion, missing diagnostics, constant/low-variance removal
      - Outlier filtering via IQR per feature (keeps NaNs)
      - Per-feature imputation (median) after filtering
      - Clipping to robust bounds to reduce residual extremes
    Returns (clean_df, diagnostics)
    """
    df = df.copy()
    # Coerce to numeric for selected features
    for col in feature_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    diagnostics = {
        "rows_before": int(len(df)),
        "missing_counts": {c: int(df[c].isna().sum()) for c in feature_cols if c in df.columns},
    }

    # Remove constant/near-constant features (very low signal)
    variances = {}
    keep_cols = []
    for c in feature_cols:
        if c in df.columns:
            var = df[c].var(skipna=True)
            variances[c] = float(var) if pd.notna(var) else 0.0
            if var is None or np.isnan(var) or var <= 1e-12:
                continue
            keep_cols.append(c)
    diagnostics["kept_features"] = keep_cols
    diagnostics["dropped_low_variance"] = [c for c in feature_cols if c in df.columns and c not in keep_cols]

    # IQR filter per kept feature
    mask = np.ones(len(df), dtype=bool)
    bounds = {}
    for col in keep_cols:
        s = df[col]
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        # If iqr is 0 (degenerate), skip filtering for this col
        if pd.isna(iqr) or iqr == 0:
            continue
        lower = q1 - iqr_factor * iqr
        upper = q3 + iqr_factor * iqr
        bounds[col] = (float(lower), float(upper))
        col_mask = s.between(lower, upper) | s.isna()
        mask &= col_mask.to_numpy()

    filtered = df.loc[mask].reset_index(drop=True)
    diagnostics["rows_after_iqr"] = int(len(filtered))

    # Impute missing values using median (robust for skewed data)
    imputer = SimpleImputer(strategy="median")
    cols_for_impute = [c for c in keep_cols if c in filtered.columns]
    if cols_for_impute:
        filtered[cols_for_impute] = imputer.fit_transform(filtered[cols_for_impute])

    # Clip to robust bounds (3 * IQR) to reduce residual extremes
    for col in cols_for_impute:
        s = filtered[col]
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        lower = q1 - 3 * iqr
        upper = q3 + 3 * iqr
        filtered[col] = s.clip(lower, upper)

    diagnostics["final_missing_counts"] = {c: int(filtered[c].isna().sum()) for c in cols_for_impute}
    diagnostics["rows_final"] = int(len(filtered))
    return filtered, diagnostics

