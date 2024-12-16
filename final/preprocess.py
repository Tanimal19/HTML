import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import csv



DATA_DIR = "data/preprocess data"
OUTPUT_DIR = "data/preprocess data"


def one_hot_encoding(df):
    """
    One-hot encoding for categorical features
    """

    cat_df = df.select_dtypes(include=['object'])
    encode_cat_df = pd.get_dummies(cat_df, dtype=float)
    df = pd.concat([df, encode_cat_df], axis=1)
    df = df.drop(cat_df.columns, axis=1)

    return df


def standard_scaler(df):
    """
    Standard scaling for numerical features
    """

    num_df = df.select_dtypes(include=['int64', 'float64'])
    num_df = num_df.drop(['season'], axis=1) # season should not scaled
    scaled_num_df = StandardScaler().fit_transform(num_df)
    df[num_df.columns] = scaled_num_df

    return df


def align_columns(df1, df2):
    """
    Align columns for two dataframes
    """

    new_df1, new_df2 = df1.align(df2, join='outer', axis=1, fill_value=0)
    assert new_df1.columns.equals(new_df2.columns)

    return new_df1, new_df2


def select_important_features(X, y, threshold):
    """
    select important features by RandomForestClassifier,
    X should be preprocessed to fit RandomForestClassifier
    ref: https://gist.github.com/Keycatowo/eb042f1fdd5dd323e0a81a0670249bfb
    """

    feature_labels = list(X.columns)

    rf = RandomForestClassifier().fit(X, y)
    importances = rf.feature_importances_ 
    indices = np.argsort(importances)[::-1]

    # with open(f"output/preprocess/feature_importance.csv", "w") as f:
    #     writer = csv.writer(f)
        
    #     writer.writerow(["rank","feature","importance"])
    #     for i in range(X.shape[1]):
    #         writer.writerow([i + 1, feature_labels[indices[i]], importances[indices[i]]])

    selected_features = []
    for i in range(X.shape[1]):
        if importances[indices[i]] > threshold:
            selected_features.append(feature_labels[indices[i]])

    print(f"Selected features: {selected_features}")

    return selected_features


def make_diff_feature(df):
    """
    make feature with 1) home>away? 2) abs(home-away)
    """

    home_columns = [col.replace("home_", "") for col in df.columns if col.startswith("home_")]
    away_columns = [col.replace("away_", "") for col in df.columns if col.startswith("away_")]

    new_df = pd.DataFrame()
    for col in home_columns:
        if col in away_columns:
            diff_column = abs(df[f"home_{col}"] - df[f"away_{col}"])
            home_better_column = (df[f"home_{col}"] > df[f"away_{col}"]).astype(int)
            new_df = pd.concat([new_df, diff_column.rename(f"diff_{col}"), home_better_column.rename(f"home_better_{col}")], axis=1)

    return new_df



train_data = pd.read_csv(f"{DATA_DIR}/nonan data/train_data_nonan.csv")
test_data = pd.read_csv(f"{DATA_DIR}/nonan data/same_season_test_data_nonan.csv")
train_data = train_data.replace({True: 1, False: 0})
test_data = test_data.replace({True: 1, False: 0})

suffix = ""

DROP_FEATURES = ["id", "home_team_season", "away_team_season"]

# scale and one-hot encoding
X = train_data.drop(["home_team_win", "date"] + DROP_FEATURES, axis=1)
X = one_hot_encoding(X)
X = standard_scaler(X)

y = train_data["home_team_win"]

X_test = test_data.drop(DROP_FEATURES, axis=1)
X_test = one_hot_encoding(X_test)
X_test = standard_scaler(X_test)

X, X_test = align_columns(X, X_test)


# remove unimportant features
if True:
    print("removing unimportant features")
    assert X.shape[1] == X_test.shape[1]

    important_features = select_important_features(X, y, 0.0005)
    X = X[important_features]
    X_test = X_test[important_features]
    suffix += "_imp"

# add diff features
if False:
    print("adding diff features")
    assert X.shape[1] == X_test.shape[1]

    X = pd.concat([X, make_diff_feature(X)], axis=1)
    X_test = pd.concat([X_test, make_diff_feature(X_test)], axis=1)
    suffix += "_withdiff"

# remove std and skew features
if False:
    print("removing std and skew features")
    assert X.shape[1] == X_test.shape[1]

    for column in X.columns:
        if column.endswith("_std") or column.endswith("_skew"):
            X = X.drop(column, axis=1)
            X_test = X_test.drop(column, axis=1)
    suffix += "_nostd"

# add weighted
if False:
    print("adding weight")
    assert X.shape[1] == X_test.shape[1]

    importance_df = pd.read_csv(f"{DATA_DIR}/analysis/feature_importance.csv")  
    for column in X.columns:
        if column in importance_df['feature'].values:
            importance = importance_df[importance_df['feature'] == column]['importance'].values[0]
            if importance == 0:
                X = X.drop(column, axis=1)
                X_test = X_test.drop(column, axis=1)
            else:
                X[column] = X[column] * importance
                X_test[column] = X_test[column] * importance
    suffix += "_weight"


# print(f"\noutput {suffix} data")
# print("X: ", X.shape)
# print("y: ", y.shape)
# print("X_test: ", X_test.shape)

# assert X.shape[1] == X_test.shape[1]

# print("\nremain columns:")
# for col in X.columns:
#     print(col)

# X.to_csv(f"{OUTPUT_DIR}/X{suffix}.csv", index=False)
# y.to_csv(f"{OUTPUT_DIR}/y{suffix}.csv", index=False)
# X_test.to_csv(f"{OUTPUT_DIR}/test_X{suffix}.csv", index=False)