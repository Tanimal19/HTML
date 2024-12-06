import pandas as pd
import numpy as np
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
import multiprocessing

import utils

TIME_STR = str(datetime.now().strftime("%m%d_%H%M"))
DATA_DIR = "data"
OUTPUT_DIR = "output/predictions"

print(f"\n--- {TIME_STR} ---\n")

train_data = pd.read_csv(f"{DATA_DIR}/train_data_N.csv")
test_data = pd.read_csv(f"{DATA_DIR}/same_season_test_data_N.csv")

# preprocess
DROP_FEATURES = ["id", "home_team_season", "away_team_season"]
X = train_data.drop(["home_team_win", "date"] + DROP_FEATURES, axis=1)
y = train_data["home_team_win"].map({True: 1, False: 0})
X_test = test_data.drop(DROP_FEATURES, axis=1)

X = utils.one_hot_encoding(X)
X = utils.standard_scaler(X)
X_test = utils.one_hot_encoding(X_test)
X_test = utils.standard_scaler(X_test)

X, X_test = utils.align_columns(X, X_test)

important_features = utils.select_important_features(X, y, 0.0005)
X = X[important_features]
X_test = X_test[important_features]


# train first layer models
print(f"[INFO]\tTraining first layer models")

# split X into train and validation set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# for categorical features, use RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

cat_X_train = X_train.select_dtypes(include=['object'])
cat_X_val = X_val.select_dtypes(include=['object'])

print(f"[LOG]\tTraining RandomForestClassifier...")
rf_start_time = time.time()
rf_model = RandomForestClassifier()
rf_model.fit(cat_X_train, y_train)
rf_pred = rf_model.predict_proba(cat_X_val)[:, 1]
print(f"[LOG]\tRandomForestClassifier training time: {time.time() - rf_start_time}")

# for numerical features, use SVC and LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

num_X_train = X_train.select_dtypes(include=['int64', 'float64'])
num_X_val = X_val.select_dtypes(include=['int64', 'float64'])

print(f"[LOG]\tTraining SVC...")
svc_start_time = time.time()
svc_model = SVC(probability=True)
svc_model.fit(num_X_train, y_train)
svc_pred = svc_model.predict_proba(num_X_val)[:, 1]
print(f"[LOG]\tSVC training time: {time.time() - svc_start_time}")

print(f"[LOG]\tTraining LogisticRegression...")
lr_start_time = time.time()
lr_model = LogisticRegression()
lr_model.fit(num_X_train, y_train)
lr_pred = lr_model.predict_proba(num_X_val)[:, 1]
print(f"[LOG]\tLogisticRegression training time: {time.time() - lr_start_time}")


# train second layer model, using GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier

print(f"[INFO]\tTraining second layer model")

blend_X_train = np.column_stack((rf_pred, svc_pred, lr_pred))
blend_y_train = y_val

second_layer_model = GradientBoostingClassifier()
second_layer_model.fit(blend_X_train, blend_y_train)


# test the model
print(f"[INFO]\tTesting the model")


blend_X_test