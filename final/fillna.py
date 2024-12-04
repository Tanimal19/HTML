import pandas as pd
import numpy as np
import time
from multiprocessing import Pool
import multiprocessing

pd.options.mode.chained_assignment = None

# preprocessing options
RANDOM_STATE = None

DATA_DIR = "_data"
OUTPUT_DIR = "output/preprocess"


TEAM_PREFIX = ["home_", "away_"]
STAT_SUFFIX = ["_mean", "_std", "_skew"]
# recent features = TEAM_PREFIX * RECENT_FEATURES
RECENT_FEATURES = [
    "batting_batting_avg_10RA",
    "batting_onbase_perc_10RA",
    "batting_onbase_plus_slugging_10RA",
    "batting_leverage_index_avg_10RA",
    "batting_RBI_10RA",
    "pitching_earned_run_avg_10RA",
    "pitching_SO_batters_faced_10RA",
    "pitching_H_batters_faced_10RA",
    "pitching_BB_batters_faced_10RA",
    "pitcher_earned_run_avg_10RA",
    "pitcher_SO_batters_faced_10RA",
    "pitcher_H_batters_faced_10RA",
    "pitcher_BB_batters_faced_10RA"
]
# seasonal features = TEAM_PREFIX * SEASONAL_FEATURES * STAT_SUFFIX
SEASONAL_FEATURES = [
    "team_errors",
    "team_spread",
    "team_wins",
    "batting_batting_avg",
    "batting_onbase_perc",
    "batting_onbase_plus_slugging",
    "batting_leverage_index_avg",
    "batting_wpa_bat",
    "batting_RBI",
    "pitching_earned_run_avg",
    "pitching_SO_batters_faced",
    "pitching_H_batters_faced",
    "pitching_BB_batters_faced",
    "pitching_leverage_index_avg",
    "pitching_wpa_def",
    "pitcher_earned_run_avg",
    "pitcher_SO_batters_faced",
    "pitcher_H_batters_faced",
    "pitcher_BB_batters_faced",
    "pitcher_leverage_index_avg",
    "pitcher_wpa_def"
]


def FEATURE_FILL_NAN(df, feature_col, group_cols, strategy, size=1, values=None):
    """
    fill NaN values in feature_col with strategy, return (feature_col, df[feature_col])

    feature_col: column to fill NaN
    group_cols: list of columns to group by, should not have NaN, if None, take the whole data as a group
    strategy:
        - "mean": fill with mean of group, only for numerical data
        - "mode": fill with mode of group
        - "nearest": fill with mean of nearest {size} value of the same group, by df["date"]
        - "random": fill with randomly select from {values}, ingore group_cols
    """

    assert feature_col in df.columns

    print(f"[LOG] \tfilling {feature_col} with {strategy}")
    start_time = time.time()

    # Keep only feature_col and group_cols in the dataframe
    keep = [feature_col] + (group_cols if group_cols is not None else [])
    if "date" in df.columns:
        keep.append("date")
    df = df[keep]


    if (strategy == "random"):
        assert values is not None

        np.random.seed(RANDOM_STATE)
        df[feature_col] = df[feature_col].map(lambda v: v if pd.notnull(v) else np.random.choice(values))
        
    else:
        # change group_cols to string to avoid error when query
        if group_cols is not None:
            original_dtype = df[group_cols].dtypes
            df[group_cols] = df[group_cols].astype(str)
        else:
            print("[WARNING] \tgroup_cols is None, take the whole data as a group")


        nan_df = df[df[feature_col].isna()]

        if (strategy == "mean"):
            if group_cols is None:
                mean = df[feature_col].mean()
                df[feature_col] = df[feature_col].fillna(mean)
            else:
                mean_df = df.groupby(group_cols)[feature_col].mean().reset_index()
                for row in nan_df.iterrows():
                    query = " & ".join([f"{col} == \"{value}\"" for col, value in zip(group_cols, row[1][group_cols])])
                    mean = mean_df.query(query)[feature_col].values[0]
                    df.loc[row[0], feature_col] = mean

        elif (strategy == "mode"):
            if group_cols is None:
                mode = df[feature_col].mode().values[0]
                df[feature_col] = df[feature_col].fillna(mode)
            else:
                mode_df = df.groupby(group_cols)[feature_col].agg(lambda v: v.mode().iloc[0]).reset_index()
                for row in nan_df.iterrows():
                    query = " & ".join([f"{col} == \"{value}\"" for col, value in zip(group_cols, row[1][group_cols])])
                    mode = mode_df.query(query)[feature_col].values[0]
                    df.loc[row[0], feature_col] = mode

        elif (strategy == "nearest"):
            assert "date" in df.columns

            for row in nan_df.iterrows():
                if group_cols is not None:
                    group_query = " & ".join([f"{col} == \"{value}\"" for col, value in zip(group_cols, row[1][group_cols])])
                    before_query = f"date < \"{row[1]['date']}\" & {group_query}"
                    after_query = f"date > \"{row[1]['date']}\" & {group_query}"
                else:
                    before_query = f"date < \"{row[1]['date']}\""
                    after_query = f"date > \"{row[1]['date']}\""

                before_rows = df.query(before_query).sort_values(by=["date"], ascending=False).dropna()
                if len(before_rows) > size:
                    before_rows = before_rows[:size]
                
                after_rows = df.query(after_query).sort_values(by=["date"]).dropna()
                if len(after_rows) > size:
                    after_rows = after_rows[:size]

                nearest_rows = pd.concat([before_rows, after_rows], ignore_index=True)
                nearest = nearest_rows[feature_col].mean()
                df.loc[row[0], feature_col] = nearest


        # restore original dtype of group_cols
        if group_cols is not None:
            df[group_cols] = df[group_cols].astype(original_dtype)


    print(f"[LOG] \t{feature_col} finished in {time.time() - start_time:.2f} seconds")

    if df[feature_col].isna().any():
        print(f"[WARNING] \t{feature_col} still has NaN values")
        
    return (feature_col, df[feature_col])


def FILL_X(input_filename):
    print(f"[LOG] \tstart fill NaN of {input_filename}")
    start_time = time.time()

    df = pd.read_csv(f"{DATA_DIR}/{input_filename}.csv")

    # fill season with date's year
    df["season"] = df["date"].map(lambda df: int(df.split("-")[0]))
    assert df["season"].isna().any() == False

    cpus = multiprocessing.cpu_count()
    pool = Pool(cpus)

    inputs = []

    # is_night_game
    inputs.append((df, "is_night_game", None, "random", 1, ["False", "True"]))

    # TEAM_pitcher
    for prefix in TEAM_PREFIX:
        inputs.append((df, prefix + "pitcher", [prefix + "team_abbr", "season"], "mode", 1, None))

    # TEAM_team_rest, TEAM_pitcher_rest
    for prefix in TEAM_PREFIX:
        for feature_name in ["team_rest", "pitcher_rest"]:
            inputs.append((df, prefix + feature_name, [prefix + "team_abbr"], "mode", 1, None))

    # TEAM_RECENT
    for prefix in TEAM_PREFIX:
        for feature_name in RECENT_FEATURES:
            inputs.append((df, prefix + feature_name, [prefix + "team_abbr"], "nearest", 3, None))

    # TEAM_SEASONAL_STAT
    for prefix in TEAM_PREFIX:
        for feature_name in SEASONAL_FEATURES:
            for suffix in STAT_SUFFIX:
                inputs.append((df, prefix + feature_name + suffix, [prefix + "team_abbr", "season"], "mean", 1, None))

    results = pool.starmap(FEATURE_FILL_NAN, inputs)
    pool.close()

    for (feature_col, feature_value) in results:
        df[feature_col] = feature_value

    df.to_csv(f"{DATA_DIR}/{input_filename}_N.csv", index=False)

    print(f"[LOG] \tall finished in {time.time() - start_time:.2f} seconds")


def FILL_X_TEST(input_filename):
    print(f"[LOG] \tstart fill NaN of {input_filename}")
    start_time = time.time()

    df = pd.read_csv(f"{DATA_DIR}/{input_filename}.csv")

    # fill season with random year range from min to max
    min_season = df["season"].min()
    max_season = df["season"].max()
    result = FEATURE_FILL_NAN(df, "season", None, "random", 1, list(range(int(min_season), int(max_season+1))))
    df["season"] = result[1]
    assert df["season"].isna().any() == False

    cpus = multiprocessing.cpu_count()
    pool = Pool(cpus)

    inputs = []

    # is_night_game
    inputs.append((df, "is_night_game", None, "random", 1, ["False", "True"]))

    # don't use season cuz it's random
    # TEAM_pitcher
    for prefix in TEAM_PREFIX:
        inputs.append((df, prefix + "pitcher", [prefix + "team_abbr"], "mode", 1, None))

    # TEAM_team_rest, TEAM_pitcher_rest
    for prefix in TEAM_PREFIX:
        for feature_name in ["team_rest", "pitcher_rest"]:
            inputs.append((df, prefix + feature_name, [prefix + "team_abbr"], "mode", 1, None))

    # TEAM_RECENT
    for prefix in TEAM_PREFIX:
        for feature_name in RECENT_FEATURES:
            inputs.append((df, prefix + feature_name, [prefix + "team_abbr"], "mean", 1, None))

    # TEAM_SEASONAL_STAT
    for prefix in TEAM_PREFIX:
        for feature_name in SEASONAL_FEATURES:
            for suffix in STAT_SUFFIX:
                inputs.append((df, prefix + feature_name + suffix, [prefix + "team_abbr"], "mean", 1, None))


    results = pool.starmap(FEATURE_FILL_NAN, inputs)
    pool.close()

    for (feature_col, feature_value) in results:
        df[feature_col] = feature_value

    df.to_csv(f"{DATA_DIR}/{input_filename}_N.csv", index=False)

    print(f"[LOG] \tall finished in {time.time() - start_time:.2f} seconds")


FILL_X("train_data")
FILL_X_TEST("same_season_test_data")
FILL_X_TEST("2024_test_data")