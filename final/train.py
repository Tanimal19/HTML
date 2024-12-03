import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import time
from datetime import datetime
from multiprocessing import Pool
import multiprocessing

pd.options.mode.chained_assignment = None


RANDOM_STATE = None
VER_NAME = str(datetime.now().strftime("%m%d_%H%M"))
DATA_DIR = "_data"
OUTPUT_DIR = "output/predictions"


def one_hot_encoding(df):
    cat_df = df.select_dtypes(include=['object'])
    encode_cat_df = pd.get_dummies(cat_df)
    df = pd.concat([df, encode_cat_df], axis=1)
    df = df.drop(cat_df.columns, axis=1)
    return df


def preprocess():
    print("start preprocessing...")
    preprocess_start_time = time.time()

    train_data = pd.read_csv(f"{DATA_DIR}/train_data_nonan.csv")
    test_data = pd.read_csv(f"{DATA_DIR}/2024_test_data_nonan.csv")
    
    DROP_FEATURES = ["id", "home_team_season", "away_team_season"]
    X = train_data.drop(["home_team_win", "date"] + DROP_FEATURES, axis=1)
    y = train_data["home_team_win"].map({True: 1, False: 0})
    X_test = test_data.drop(DROP_FEATURES, axis=1)

    assert X.isna().any().any() == False
    assert X_test.isna().any().any() == False

    # one-hot encoding
    X = one_hot_encoding(X)
    X_test = one_hot_encoding(X_test)

    # align columns
    X_columns = set(X.columns)
    X_test_columns = set(X_test.columns)
    full_col = X_columns.union(X_test_columns)

    X_missing = pd.DataFrame(0, index=X.index, columns=list(full_col - X_columns))
    X = pd.concat([X, X_missing], axis=1)

    X_test_missing = pd.DataFrame(0, index=X_test.index, columns=list(full_col - X_test_columns))
    X_test = pd.concat([X_test, X_test_missing], axis=1)

    X = X[X_test.columns]

    print("X shape:", X.shape)
    print("X_test shape:", X_test.shape)

    assert X.isna().any().any() == False
    assert X_test.isna().any().any() == False

    print("\npreprocessing time %ss" % (time.time() - preprocess_start_time))

    return X, y, X_test


def train(X, y):
    print("start training...")
    train_start_time = time.time()

    decay_factor = 5
    weights = np.exp(-(2024 - X["season"]) / decay_factor)

    model = RandomForestClassifier(random_state=RANDOM_STATE)
    model.fit(X, y)

    print("\ntraining time %ss" % (time.time() - train_start_time))

    return model


def predict(model, X_test):
    print("start testing...")
    test_start_time = time.time()

    y_pred = model.predict(X_test)

    index = list(range(0,len(y_pred)))
    df = pd.DataFrame({"id": index, "home_team_win": y_pred})
    df["home_team_win"] = df["home_team_win"].map({1: True, 0: False})

    print("\ntest time %ss" % (time.time() - test_start_time))

    return df


print(f"\n\n--- ver.{VER_NAME} ---\n")

X, y, X_test = preprocess()
model = train(X, y)
result = predict(model, X_test)

result.to_csv(f"{OUTPUT_DIR}/predictions_{VER_NAME}.csv", index=False)