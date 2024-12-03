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


# Main
X = pd.read_csv(f"{DATA_DIR}/train_data.csv")

FILLNA_X = False
if FILLNA_X:
    print("--- start fill NaN of X ---")
    start_time = time.time()

    # fill season with date's year
    X["season"] = X["date"].map(lambda x: int(x.split("-")[0]))
    assert X["season"].isna().any() == False


    # For is_night_game
    # change to 0 and 1
    # fill NaN with random 0 or 1
    FILLNA_IS_NIGHT_GAME = True
    if FILLNA_IS_NIGHT_GAME:
        print("filling is_night_game")

        np.random.seed(RANDOM_STATE)
        is_night_game = X["is_night_game"].map({True: 1, False: 0})
        is_night_game = is_night_game.map(lambda X: X if pd.notnull(X) else np.random.choice([0, 1]))
        X["is_night_game"] = is_night_game

        assert X["is_night_game"].isna().any() == False


    # For home_pitcher, away_pitcher
    # fill NaN with the most common pitcher in each team & season
    FILLNA_PITCHER = True
    if FILLNA_PITCHER:
        print("filling pitcher")

        season_col = X["season"]
        double_season_col = pd.concat([season_col, season_col], ignore_index=True)
        team_col = pd.concat([X["home_team_abbr"], X["away_team_abbr"]], ignore_index=True)
        pitcher = pd.concat([X["home_pitcher"], X["away_pitcher"]], ignore_index=True)
        pitcher = pitcher.fillna("unknown")
        team_pitcher = pd.concat([double_season_col, team_col, pitcher], axis=1)
        team_pitcher.columns = ["season", "team", "pitcher"]

        team_pitcher_counts = team_pitcher.value_counts().reset_index(name='count')
        team_pitcher_counts = team_pitcher_counts.sort_values(by=['team', 'season', 'pitcher'])
        # team_pitcher_counts.to_csv(f"{OUTPUT_DIR}/team_pitcher_counts.csv", index=False)

        most_common_pitcher = team_pitcher_counts.loc[team_pitcher_counts.groupby(['team', 'season'])['count'].idxmax()]
        # most_common_pitcher.to_csv(f"{OUTPUT_DIR}/most_common_pitcher.csv", index=False)

        unknown_pitcher = team_pitcher[team_pitcher["pitcher"] == "unknown"]
        for row in unknown_pitcher.iterrows():
            season, team = row[1]["season"], row[1]["team"]
            most_common = most_common_pitcher[(most_common_pitcher["team"] == team) & (most_common_pitcher["season"] == season)]["pitcher"].values[0]
            team_pitcher.loc[row[0], "pitcher"] = most_common

        X["home_pitcher"] = team_pitcher.iloc[:len(X)]["pitcher"].values
        X["away_pitcher"] = team_pitcher.iloc[len(X):]["pitcher"].values

        assert X["home_pitcher"].isna().any() == False
        assert X["away_pitcher"].isna().any() == False


    # For team_rest, pitcher_rest
    # fill NaN with most common value among all data
    FILLNA_REST = True
    if FILLNA_REST:
        print("filling rest")

        rest_cols = X.columns[X.columns.str.contains("_rest")]
        for col in rest_cols:
            most_common = X[col].mode().values[0]
            X[col] = X[col].fillna(most_common)

        assert X[rest_cols].isna().any().any() == False


    # For seasonal features
    # fill NaN with mean of team & season
    FILLNA_SEASON = True    
    if FILLNA_SEASON:
        season_col = X["season"]
        double_season_col = pd.concat([season_col, season_col], ignore_index=True)
        team_col = pd.concat([X["home_team_abbr"], X["away_team_abbr"]], ignore_index=True)

        for column_name in SEASONAL_FEATURES:
            for suffix in STAT_SUFFIX:
                print(f"filling {column_name + suffix}")

                home_stats_name = TEAM_PREFIX[0] + column_name + suffix
                away_stats_name = TEAM_PREFIX[1] + column_name + suffix

                stats_col = pd.concat([X[home_stats_name], X[away_stats_name]], ignore_index=True)
                team_stats = pd.concat([double_season_col, team_col, stats_col], axis=1)
                team_stats.columns = ["season", "team", "stats"]

                team_stats_mean = team_stats.groupby(['team', 'season']).mean().reset_index()

                nan_team_stats = team_stats[team_stats["stats"].isna()]
                for row in nan_team_stats.iterrows():
                    season, team = row[1]["season"], row[1]["team"]
                    mean = team_stats_mean[(team_stats_mean["team"] == team) & (team_stats_mean["season"] == season)]["stats"].values[0]
                    team_stats.loc[row[0], "stats"] = mean

                X[home_stats_name] = team_stats.iloc[:len(X)]["stats"].values
                X[away_stats_name] = team_stats.iloc[len(X):]["stats"].values

                assert X[home_stats_name].isna().any() == False
                assert X[away_stats_name].isna().any() == False


    # For recent features
    # fill NAN with nearest (by date) valid value of the same team
    FILLNA_RECENT = True
    if FILLNA_RECENT:
        date_col = X["date"]
        double_date_col = pd.concat([date_col, date_col], ignore_index=True)
        team_col = pd.concat([X["home_team_abbr"], X["away_team_abbr"]], ignore_index=True)

        for column_name in RECENT_FEATURES:
            print(f"filling {column_name}")

            home_stats_name = TEAM_PREFIX[0] + column_name
            away_stats_name = TEAM_PREFIX[1] + column_name

            stats_col = pd.concat([X[home_stats_name], X[away_stats_name]], ignore_index=True)
            team_stats = pd.concat([double_date_col, team_col, stats_col], axis=1)
            team_stats.columns = ["date", "team", "stats"]
            team_stats = team_stats.sort_values(by=["date", "team"])

            nan_team_stats = team_stats[team_stats["stats"].isna()]

            for row in nan_team_stats.iterrows():
                date, team = row[1]["date"], row[1]["team"]

                # if no any data before the date, find the nearest data after
                before_rows = team_stats[(team_stats["team"] == team) & (team_stats["date"] < date)].dropna()
                if before_rows.empty:
                    after_rows = team_stats[(team_stats["team"] == team) & (team_stats["date"] > date)].dropna()
                    nearest = after_rows.iloc[:1]["stats"].values[0]
                else:
                    nearest = before_rows.iloc[-1:]["stats"].values[0]

                team_stats.loc[row[0], "stats"] = nearest

            team_stats = team_stats.sort_index()

            X[home_stats_name] = team_stats.iloc[:len(X)]["stats"].values
            X[away_stats_name] = team_stats.iloc[len(X):]["stats"].values

            assert X[home_stats_name].isna().any() == False
            assert X[away_stats_name].isna().any() == False


    # home_team_season, away_team_season may have NaN values
    X.to_csv(f"{DATA_DIR}/train_data_nonan.csv", index=False)

    print(f"--- finished in {time.time() - start_time:.2f} seconds ---")


X_test = pd.read_csv(f"{DATA_DIR}/2024_test_data.csv")

FILLNA_X_TEST = True
if FILLNA_X_TEST:
    print("--- start fill NaN of X_test ---")
    start_time = time.time()

    # fill season with random year range from min to max
    min_season = X_test["season"].min()
    max_season = X_test["season"].max()
    np.random.seed(RANDOM_STATE)
    X_test["season"] = X_test["season"].map(lambda x: np.random.randint(min_season, max_season+1))
    assert X_test["season"].isna().any() == False


    # For is_night_game
    # change to 0 and 1
    # fill NaN with random 0 or 1
    FILLNA_IS_NIGHT_GAME = True
    if FILLNA_IS_NIGHT_GAME:
        print("filling is_night_game")

        np.random.seed(RANDOM_STATE)
        is_night_game = X_test["is_night_game"].map({True: 1, False: 0})
        is_night_game = is_night_game.map(lambda X_test: X_test if pd.notnull(X_test) else np.random.choice([0, 1]))
        X_test["is_night_game"] = is_night_game

        assert X_test["is_night_game"].isna().any() == False


    # For home_pitcher, away_pitcher
    # fill NaN with the most common pitcher in each team & season
    FILLNA_PITCHER = True
    if FILLNA_PITCHER:
        print("filling pitcher")

        season_col = X_test["season"]
        double_season_col = pd.concat([season_col, season_col], ignore_index=True)
        team_col = pd.concat([X_test["home_team_abbr"], X_test["away_team_abbr"]], ignore_index=True)
        pitcher = pd.concat([X_test["home_pitcher"], X_test["away_pitcher"]], ignore_index=True)
        pitcher = pitcher.fillna("unknown")
        team_pitcher = pd.concat([double_season_col, team_col, pitcher], axis=1)
        team_pitcher.columns = ["season", "team", "pitcher"]

        team_pitcher_counts = team_pitcher.value_counts().reset_index(name='count')
        team_pitcher_counts = team_pitcher_counts.sort_values(by=['team', 'season', 'pitcher'])
        # team_pitcher_counts.to_csv(f"{OUTPUT_DIR}/team_pitcher_counts.csv", index=False)

        most_common_pitcher = team_pitcher_counts.loc[team_pitcher_counts.groupby(['team', 'season'])['count'].idxmax()]
        # most_common_pitcher.to_csv(f"{OUTPUT_DIR}/most_common_pitcher.csv", index=False)

        unknown_pitcher = team_pitcher[team_pitcher["pitcher"] == "unknown"]
        for row in unknown_pitcher.iterrows():
            season, team = row[1]["season"], row[1]["team"]
            most_common = most_common_pitcher[(most_common_pitcher["team"] == team) & (most_common_pitcher["season"] == season)]["pitcher"].values[0]
            team_pitcher.loc[row[0], "pitcher"] = most_common

        X_test["home_pitcher"] = team_pitcher.iloc[:len(X_test)]["pitcher"].values
        X_test["away_pitcher"] = team_pitcher.iloc[len(X_test):]["pitcher"].values

        assert X_test["home_pitcher"].isna().any() == False
        assert X_test["away_pitcher"].isna().any() == False


    # For team_rest, pitcher_rest
    # fill NaN with most common value among all data
    FILLNA_REST = True
    if FILLNA_REST:
        print("filling rest")

        rest_cols = X_test.columns[X_test.columns.str.contains("_rest")]
        for col in rest_cols:
            most_common = X_test[col].mode().values[0]
            X_test[col] = X_test[col].fillna(most_common)

        assert X_test[rest_cols].isna().any().any() == False


    # For seasonal & recent features
    # fill NaN with mean of team & season
    FILLNA_SEASON_RECENT = True    
    if FILLNA_SEASON_RECENT:
        season_col = X_test["season"]
        double_season_col = pd.concat([season_col, season_col], ignore_index=True)
        team_col = pd.concat([X_test["home_team_abbr"], X_test["away_team_abbr"]], ignore_index=True)

        column_names = []
        for name in SEASONAL_FEATURES:
            for suffix in STAT_SUFFIX:
                column_names.append(name + suffix)
        for name in RECENT_FEATURES:
            column_names.append(name)
        
        for column_name in column_names:
            print(f"filling {column_name}")

            home_stats_name = TEAM_PREFIX[0] + column_name
            away_stats_name = TEAM_PREFIX[1] + column_name

            stats_col = pd.concat([X_test[home_stats_name], X_test[away_stats_name]], ignore_index=True)
            team_stats = pd.concat([double_season_col, team_col, stats_col], axis=1)
            team_stats.columns = ["season", "team", "stats"]

            team_stats_mean = team_stats.groupby(['team', 'season']).mean().reset_index()

            nan_team_stats = team_stats[team_stats["stats"].isna()]
            for row in nan_team_stats.iterrows():
                season, team = row[1]["season"], row[1]["team"]
                mean = team_stats_mean[(team_stats_mean["team"] == team) & (team_stats_mean["season"] == season)]["stats"].values[0]
                team_stats.loc[row[0], "stats"] = mean

            X_test[home_stats_name] = team_stats.iloc[:len(X_test)]["stats"].values
            X_test[away_stats_name] = team_stats.iloc[len(X_test):]["stats"].values

            assert X_test[home_stats_name].isna().any() == False
            assert X_test[away_stats_name].isna().any() == False


    # home_team_season, away_team_season may have NaN values
    X_test.to_csv(f"{DATA_DIR}/2024_test_data_nonan.csv", index=False)

    print(f"--- finished in {time.time() - start_time:.2f} seconds ---")