"""quickstart_xgboost: A Flower / XGBoost app."""

import numpy as np
import xgboost as xgb
from flwr.app import ArrayRecord, Context
from flwr.common.config import unflatten_dict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedXgbBagging
from challengcsibalance.lib.task import replace_keys
from xgboost import XGBRegressor

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None: 
    print("Starting Flower XGBoost server...")
    print(context.run_config)
    # Read run config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_train = context.run_config["fraction-train"]
    fraction_evaluate = context.run_config["fraction-evaluate"]
    # Flatted config dict and replace "-" with "_"
    cfg = replace_keys(unflatten_dict(context.run_config))
    params = cfg["params"]

    # Init global model
    # Init with an empty object; the XGBooster will be created
    # and trained on the client side.
    global_model = b""
    # Note: we store the model as the first item in a list into ArrayRecord,
    # which can be accessed using index ["0"].
    arrays = ArrayRecord([np.frombuffer(global_model, dtype=np.uint8)])

    # Initialize FedXgbBagging strategy
    strategy = FedXgbBagging(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
    )

    # # Start strategy, run FedXgbBagging for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
    )

    # # Save final model to disk
    global_model = bytearray(result.arrays["0"].numpy().tobytes())

    # Load aggregated model
    booster = xgb.Booster()
    booster.load_model(global_model)


    # X_train, y_train, X_val, y_val = load_test_data()
    # dtest=xgb.DMatrix(X_val, label=y_val)
    # preds = booster.predict(dtest)
    # from sklearn.metrics import mean_absolute_error
    # mae = mean_absolute_error(y_val, preds)
    # print(f"Final MAE on validation set: {mae}")

    booster.save_model("final_model.json")





























import pandas as pd
import numpy as np
import re
from pathlib import Path
import os
import ast
from challengcsibalance.lib.task import convert_timeseries_columns, clean_ts

def load_test_data():
    base = Path('./exploration/filtered_data/')

    groups_dfs = {}


    for csv_path in sorted(base.glob(f'group*/*.csv')):
        group = csv_path.parent.name
        m = re.search(r'dataset_user_(\d+)_train\.csv', csv_path.name)
        if not m:
            continue
        user_id = int(m.group(1))
        df = pd.read_csv(csv_path)
        groups_dfs.setdefault(group, {})[user_id] = df


    for group, user_dfs in groups_dfs.items():
        for user_id, df in user_dfs.items():
            groups_dfs[group][user_id] = convert_timeseries_columns(df)
    
    for group, user_dfs in groups_dfs.items():
        for user_id, df in user_dfs.items():
            for col in df.columns:
                if col.startswith('timeseries_'):
                    df[col] = df[col].apply(lambda x: clean_ts(x))
            groups_dfs[group][user_id] = df
    

    for group, users in groups_dfs.items():
        for user_id, df in users.items():
            df['deep_sleep_pct'] = np.nan
            df['rem_sleep_pct'] = np.nan
            df['light_sleep_pct'] = np.nan
            df['awake_sleep_pct'] = np.nan
            df['sleep_efficiency'] = np.nan
            df['sleep_day_ptc'] = np.nan
            for row_idx, row in df.iterrows():
                sleep_deepSleepSeconds = row.get("sleep_deepSleepSeconds", np.nan)
                sleep_lightSleepSeconds = row.get("sleep_lightSleepSeconds", np.nan)
                sleep_remSleepSeconds = row.get("sleep_remSleepSeconds", np.nan)
                sleep_timeSeconds = row.get("sleep_sleepTimeSeconds", np.nan)
                sleep_awakeTimeSeconds = row.get("sleep_awakeSleepSeconds", np.nan)
                


                deep_pct = sleep_deepSleepSeconds / sleep_timeSeconds 
                rem_pct = sleep_remSleepSeconds / sleep_timeSeconds
                light_pct = sleep_lightSleepSeconds / sleep_timeSeconds
                awake_pct = sleep_awakeTimeSeconds / sleep_timeSeconds
                sleep_efficiency = sleep_timeSeconds / (sleep_timeSeconds + sleep_awakeTimeSeconds)
                df.at[row_idx, 'deep_sleep_pct'] = deep_pct
                df.at[row_idx, 'rem_sleep_pct'] = rem_pct
                df.at[row_idx, 'light_sleep_pct'] = light_pct
                df.at[row_idx, 'awake_sleep_pct'] = awake_pct
                df.at[row_idx, 'sleep_efficiency'] = sleep_efficiency
                df.at[row_idx, 'sleep_day_ptc'] = sleep_timeSeconds / 86400.0
            
            groups_dfs[group][user_id] = df
    FEATURES = [
        # sleep composition
        "deep_sleep_pct",
        "rem_sleep_pct",
        "light_sleep_pct",
        "awake_sleep_pct",
        "sleep_efficiency",

        # heart rate static
        "hr_restingHeartRate",
        "hr_lastSevenDaysAvgRestingHeartRate",
        "hr_maxHeartRate",
        "hr_minHeartRate",

        # stress static
        "str_avgStressLevel",
        "str_maxStressLevel",

        # activity
        "act_totalCalories",
        "act_activeKilocalories",
        "act_distance",

        # respiration static
        "resp_lowestRespirationValue",
        "resp_highestRespirationValue",
        "resp_avgSleepRespirationValue",

        'sleep_day_ptc'
    ]

    X_list = []
    y_list = []

    for group, users in groups_dfs.items():
        for user_id, df in users.items():

            df_model = df.copy()

            # tieni solo righe valide
            df_model = df_model.dropna(subset=["label"])

            X = df_model[FEATURES]
            y = df_model["label"]

            X_list.append(X)
            y_list.append(y)

    X_all = pd.concat(X_list, axis=0)
    y_all = pd.concat(y_list, axis=0)

    X_all = X_all.replace([np.inf, -np.inf], np.nan)

    split_idx = int(len(X_all) * 0.7)



    X_train = X_all.iloc[:split_idx]
    X_val   = X_all.iloc[split_idx:]



    y_train = y_all.iloc[:split_idx]
    y_val   = y_all.iloc[split_idx:]
    return X_train, y_train, X_val, y_val