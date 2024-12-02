import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time
from datetime import datetime

pd.options.mode.chained_assignment = None

RANDOM_STATE = 42
TRAIN_METHOD = "svc_rbf"
DATA_DIR = "_data"
OUTPUT_DIR = "output"

def fill_nan(df):
    df["is_night_game"] = df["is_night_game"].astype(bool)

    # numerical columns - fill with linear interpolation, ffill, bfill and standardize
    num_df = df[df.select_dtypes(include=['int64', 'float64']).columns]
    num_df = num_df.interpolate(method="linear")
    num_df = num_df.ffill().bfill()

    scaler = StandardScaler()
    num_df[num_df.columns] = scaler.fit_transform(num_df)

    # categorical columns - fill with "missing" and one-hot encoding
    cat_df = df[df.select_dtypes(include=['object']).columns]
    cat_df = cat_df.fillna("missing")
    cat_df = pd.get_dummies(cat_df, dtype=float)

    # boolean columns - randomly select 0 or 1
    bool_df = df[df.select_dtypes(include=['bool']).columns]
    bool_df = bool_df.map({True: 1, False: 0})
    bool_df = bool_df.map(lambda x: x if pd.notnull(x) else bool(pd.np.random.choice([0, 1])))

    df = pd.concat([bool_df, num_df, cat_df], axis=1)
    return df

def fill_missing_column(df1, df2):
    df1_columns = set(df1.columns)
    df2_columns = set(df2.columns)
    full_col = df1_columns.union(df2_columns)

    df1_missing = pd.DataFrame(0, index=df1.index, columns=list(full_col - df1_columns))
    full_df1 = pd.concat([df1, df1_missing], axis=1)

    df2_missing = pd.DataFrame(0, index=df2.index, columns=list(full_col - df2_columns))
    full_df2 = pd.concat([df2, df2_missing], axis=1)

    full_df1 = full_df1[full_df2.columns]

    return full_df1, full_df2

def train(X, y):
    print("start training...")
    train_start_time = time.time()
    
    model = SVC(kernel="rbf", gamma='auto', random_state=RANDOM_STATE)
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


train_data = pd.read_csv(f"{DATA_DIR}/train_data.csv")
X_test = pd.read_csv(f"{DATA_DIR}/same_season_test_data.csv")

# Preprocess
print("start preprocessing...")
DROP_FEATURES = ["id"]
X = train_data.drop(["home_team_win", "date"] + DROP_FEATURES, axis=1)
X_test = X_test.drop(DROP_FEATURES, axis=1)
y = train_data["home_team_win"].map({True: 1, False: 0})

X = fill_nan(X)
X_test = fill_nan(X_test)
X, X_test = fill_missing_column(X, X_test)

print(X)
print(y)
print(X_test)


# model = train(X, y)
# result = predict(model, X_test)

# current_time = datetime.now().strftime("%m%d_%H%M")
# result.to_csv(f"{OUTPUT_DIR}/predictions_{TRAIN_METHOD}_{current_time}.csv", index=False)