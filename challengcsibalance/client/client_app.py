"""quickstart-xgboost: A Flower / XGBoost app."""

import warnings

import numpy as np
import xgboost as xgb
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common.config import unflatten_dict
from challengcsibalance.lib.task import load_data, clean_ts, convert_timeseries_columns, replace_keys
warnings.filterwarnings("ignore", category=UserWarning)
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Flower ClientApp
app = ClientApp()


def _local_boost(bst_input, num_local_round, train_dmatrix):
    # Update trees based on local training data.
    for i in range(num_local_round):
        bst_input.update(train_dmatrix, bst_input.num_boosted_rounds())

    # Bagging: extract the last N=num_local_round trees for sever aggregation
    bst = bst_input[
        bst_input.num_boosted_rounds()
        - num_local_round : bst_input.num_boosted_rounds()
    ]
    return bst


@app.train()
def train(msg: Message, context: Context) -> Message:
    
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    print('partiton ', partition_id, num_partitions)

    
    
    groups_dfs = load_data(partition_id, num_partitions)
    
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

    split_idx = int(len(X_all) * 1)



    X_train = X_all.iloc[:split_idx]
    X_val   = X_all.iloc[split_idx:]



    y_train = y_all.iloc[:split_idx]
    y_val   = y_all.iloc[split_idx:]

    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )

    # Training semplice

    global_round = msg.content["config"]["server-round"]
    if global_round == 1:
        model.fit(
            X_train,
            y_train
        )
    else:
        global_model = bytearray(msg.content["arrays"]["0"].numpy().tobytes())

        # Load global model into booster
        model.load_model(global_model)

        # Local training
        model.fit(
            X_train,
            y_train
        )
    # Save model   
    booster = model.get_booster()
    local_model = booster.save_raw("json")

    model_np = np.frombuffer(local_model, dtype=np.uint8)
    # Construct reply message
    # Note: we store the model as the first item in a list into ArrayRecord,
    # which can be accessed using index ["0"].
    model_record = ArrayRecord([model_np])
    metrics = {
        "num-examples": len(X_train),
    }
    metric_record = MetricRecord(metrics)

    content = RecordDict({"arrays": model_record, "metrics": metric_record})


    return Message(content=content, reply_to=msg)



    # # Load model and data
    # partition_id = context.node_config["partition-id"]
    # num_partitions = context.node_config["num-partitions"]
    # train_dmatrix, _, num_train, _ = load_data(partition_id, num_partitions)

    # # Read from run config
    # num_local_round = context.run_config["local-epochs"]
    # # Flatted config dict and replace "-" with "_"
    # cfg = replace_keys(unflatten_dict(context.run_config))
    # params = cfg["params"]

    # global_round = msg.content["config"]["server-round"]
    # if global_round == 1:
    #     # First round local training
    #     bst = xgb.train(
    #         params,
    #         train_dmatrix,
    #         num_boost_round=num_local_round,
    #     )
    # else:
    #     bst = xgb.Booster(params=params)
    #     global_model = bytearray(msg.content["arrays"]["0"].numpy().tobytes())

    #     # Load global model into booster
    #     bst.load_model(global_model)

    #     # Local training
    #     bst = _local_boost(bst, num_local_round, train_dmatrix)

    # # Save model
    # local_model = bst.save_raw("json")
    # model_np = np.frombuffer(local_model, dtype=np.uint8)

    # # Construct reply message
    # # Note: we store the model as the first item in a list into ArrayRecord,
    # # which can be accessed using index ["0"].
    # model_record = ArrayRecord([model_np])
    # metrics = {
    #     "num-examples": num_train,
    # }
    # metric_record = MetricRecord(metrics)
    # content = RecordDict({"arrays": model_record, "metrics": metric_record})
    # return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:


    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    groups_dfs = load_data(partition_id, num_partitions)
    
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

    cfg = replace_keys(unflatten_dict(context.run_config))
    params = cfg["params"]

    bst = XGBRegressor(
        params=params
    )
    global_model = bytearray(msg.content["arrays"]["0"].numpy().tobytes())
    # Load global model into booster
    bst.load_model(global_model)
    # Run evaluation
    auc = bst.score(
        X_val,
        y_val
    )

    # Construct and return reply Message
    metrics = {
        "auc": auc,
        "num-examples": len(X_val),
        'mae': mean_absolute_error(y_val, bst.predict(X_val)),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)


    # # Load model and data
    # partition_id = context.node_config["partition-id"]
    # num_partitions = context.node_config["num-partitions"]
    # _, valid_dmatrix, _, num_val = load_data(partition_id, num_partitions)

    # # Load config
    # cfg = replace_keys(unflatten_dict(context.run_config))
    # params = cfg["params"]

    # # Load global model
    # bst = xgb.Booster(params=params)
    # global_model = bytearray(msg.content["arrays"]["0"].numpy().tobytes())
    # bst.load_model(global_model)

    # # Run evaluation
    # eval_results = bst.eval_set(
    #     evals=[(valid_dmatrix, "valid")],
    #     iteration=bst.num_boosted_rounds() - 1,
    # )
    # auc = float(eval_results.split("\t")[1].split(":")[1])

    # # Construct and return reply Message
    # metrics = {
    #     "auc": auc,
    #     "num-examples": num_val,
    # }
    # metric_record = MetricRecord(metrics)
    # content = RecordDict({"metrics": metric_record})
    # return Message(content=content, reply_to=msg)