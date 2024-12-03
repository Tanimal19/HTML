import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time
from datetime import datetime
from multiprocessing import Pool
import multiprocessing

pd.options.mode.chained_assignment = None


RANDOM_STATE = None
VER_NAME = str(datetime.now().strftime("%m%d_%H%M"))
DATA_DIR = "_data"
OUTPUT_DIR = "output"


def fill_nan(df):
    df["is_night_game"] = df["is_night_game"].astype(bool)

    # numerical columns - fill with mean and standardize
    num_df = df[df.select_dtypes(include=['int64', 'float64']).columns]
    imputer = SimpleImputer(strategy='mean')
    num_df[num_df.columns] = imputer.fit_transform(num_df)
    scaler = StandardScaler()
    num_df[num_df.columns] = scaler.fit_transform(num_df)

    # categorical columns - fill with "missing" and one-hot encoding
    cat_df = df[df.select_dtypes(include=['object']).columns]
    cat_df = cat_df.fillna("missing")
    cat_df = pd.get_dummies(cat_df, dtype=float)

    # boolean columns - randomly select 0 or 1
    bool_df = df[df.select_dtypes(include=['bool']).columns]
    for col in bool_df.columns:
        bool_df[col] = bool_df[col].map({True: 1, False: 0})
        bool_df[col] = bool_df[col].map(lambda x: x if pd.notnull(x) else bool(pd.np.random.choice([0, 1])))

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

def preprocess():
    print("start preprocessing...")
    preprocess_start_time = time.time()

    train_data = pd.read_csv(f"{DATA_DIR}/train_data.csv")
    X_test = pd.read_csv(f"{DATA_DIR}/same_season_test_data.csv")
    
    DROP_FEATURES = ["id"]
    X = train_data.drop(["home_team_win", "date"] + DROP_FEATURES, axis=1)
    X_test = X_test.drop(DROP_FEATURES, axis=1)
    y = train_data["home_team_win"].map({True: 1, False: 0})

    X = fill_nan(X)
    X_test = fill_nan(X_test)
    X, X_test = fill_missing_column(X, X_test)

    print("\npreprocessing time %ss" % (time.time() - preprocess_start_time))

    return X, y, X_test

def train_task(kernel, gamma):
    print(f"\nstart training... kernel: {kernel}, gamma: {gamma}")
    task_start_time = time.time()

    model = SVC(kernel=kernel, gamma=gamma, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    score = accuracy_score(y_val, y_pred)

    print(f"\nkernel: {kernel}, gamma: {gamma}, score: {score}, time: {time.time() - task_start_time}s")

    return model, score

def train(X, y):
    print("start training...")
    train_start_time = time.time()

    global X_train, X_val, y_train, y_val;
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    kernel_list = ['linear', 'poly', 'rbf']
    gamma_list = ['scale', 0.01, 0.1]

    cpus = multiprocessing.cpu_count()
    inputs = []
    for kernel in kernel_list:
        for gamma in gamma_list:
            inputs.append((kernel, gamma))
    
    p = Pool(cpus if len(inputs) > cpus else len(inputs))
    results = p.starmap(train_task, inputs)
    
    best_score = -1
    best_model = None
    best_kernel = ""
    best_gamma = ""

    for (model, score), (kernel, gamma) in zip(results, inputs):
        if score > best_score:
            best_score = score
            best_model = model
            best_kernel = kernel
            best_gamma = gamma

    return best_model


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


def preprocess(param):
    idx, X, pitcherAll = param
    numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
    home_columns = [col for col in numerical_cols if col.startswith("home_")]
    away_columns = [col for col in numerical_cols if col.startswith("away_")]

    diff = np.zeros((np.array(X).shape[0], 1))
    diff_raw = np.zeros((np.array(X).shape[0], 1))

    for home_col, away_col in zip(home_columns, away_columns):
        diff_raw = np.hstack((diff_raw, np.array(X[home_col] - X[away_col]).reshape(-1, 1)))

    N = diff_raw.shape[0]

    temp = diff_raw[:, 1:]
    temp[temp > 0] = 1
    temp[temp <= 0] = -1
    diff = np.hstack((diff, temp))
    for i in range(-1, 1):
        temp = diff_raw[:, 1:]
        temp[temp > 10 ** i] = 1
        temp[temp <= 10 ** i] = -1
        diff = np.hstack((diff, temp))

    diff = diff[:, 1:]
    d_numer = diff.shape[1]

    is_night_game = np.array(X["is_night_game"]) * 2 - 1
    diff = np.hstack((diff, is_night_game.reshape(-1, 1)))

    home_team_abbr = np.array(X["home_team_abbr"]).reshape(-1, 1)
    away_team_abbr = np.array(X["away_team_abbr"]).reshape(-1, 1)
    teamAll_np = np.hstack((home_team_abbr, away_team_abbr))
    teamAll = sorted(list(set(teamAll_np.flatten())))
    for i in range(len(teamAll)):
        temp = teamAll_np[:, :]
        temp[temp != teamAll[i]] = -1
        temp[temp == teamAll[i]] = 1
        diff = np.hstack((diff, temp))

    d_team = diff.shape[1] - d_numer

    home_pitcher = np.array(X["home_pitcher"]).reshape(-1, 1)
    away_pitcher = np.array(X["away_pitcher"]).reshape(-1, 1)
    pitcherAll_np = np.hstack((home_pitcher, away_pitcher))
    pitcherAll_np[pitcherAll_np != pitcherAll_np] = "unknown"
    for i in range(len(pitcherAll)):
        if pitcherAll[i] == "unknown":
            continue
        temp = pitcherAll_np[:, :] 
        temp[temp != pitcherAll[i]] = -1
        temp[temp == pitcherAll[i]] = 1
        diff = np.hstack((diff, temp))

    d_pitcher = diff.shape[1] - d_team
    return (idx, d_numer, d_team, d_pitcher, diff)