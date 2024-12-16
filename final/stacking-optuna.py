import pandas as pd
import numpy as np

import os
import time
from datetime import datetime
import multiprocessing
from multiprocessing import Pool

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import optuna


DATA_DIR = "data/preprocess data"
OUTPUT_DIR = "predictions"
START_TIME = str(datetime.now().strftime("%m%d_%H%M"))

DATA_SUFFIX = "_imp_withdiff"

X = pd.read_csv(f"{DATA_DIR}/X{DATA_SUFFIX}.csv")
y = pd.read_csv(f"{DATA_DIR}/y.csv")
y = y.values.ravel()
X_test = pd.read_csv(f"{DATA_DIR}/test_X{DATA_SUFFIX}.csv")


# sample weight by season
sample_weight = 


print("X: ", X.shape)
print("y: ", y.shape)
print("X_test: ", X_test.shape)

RANDOM_STATE = None
VERBOSE = 1
KFOLD = 5

global X_train, X_val, y_train, y_val
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)


def objective(trial):
    print(f"start trial: {trial.number}")

    max_depth = 10
    n_estimators = 288
    learning_rate = trial.suggest_float('learning_rate', 0.015, 0.3)
    C = trial.suggest_float('C', 0.1, 10, log=True)

    base_models = [
        ('dtc', DecisionTreeClassifier(max_depth=max_depth)),
        ('rf', RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)),
        ('etc', ExtraTreesClassifier(max_depth=max_depth, n_estimators=n_estimators)),
        ('gbc', GradientBoostingClassifier(max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate)),
        ('svc', SVC(probability=True, C=C)),
    ]
    
    meta_C = trial.suggest_float('meta_C', 0.1, 10, log=True)
    meta_model = LogisticRegression(C=meta_C, random_state=RANDOM_STATE)
    
    # start training
    stacking_clf = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=KFOLD)    
    stacking_clf.fit(X_train, y_train)
    
    y_pred = stacking_clf.predict(X_val)
    auc = roc_auc_score(y_val, y_pred)
    
    return auc


print(f"\n--- {START_TIME} ---\n")
start_time = time.time()

# start optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20, n_jobs=20)

print("best parameters:", study.best_params)
print("best AUC:", study.best_value)

# train with best parameters
max_depth = 10
n_estimators = 288
learning_rate = study.best_params['learning_rate']
C = study.best_params['C']

base_models = [
    ('dtc', DecisionTreeClassifier(max_depth=max_depth)),
    ('rf', RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)),
    ('etc', ExtraTreesClassifier(max_depth=max_depth, n_estimators=n_estimators)),
    ('gbc', GradientBoostingClassifier(max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate)),
    ('svc', SVC(probability=True, C=C)),
]

meta_C = study.best_params['meta_C']
meta_model = LogisticRegression(C=meta_C, random_state=RANDOM_STATE)

print("\nstart training")
stacking_clf = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=KFOLD)
stacking_clf.fit(X_train, y_train)

# predict test data
print("prediction:")
pred = stacking_clf.predict(X_test)
pred = pred > 0.5

pred_df = pd.DataFrame({'id': range(0, len(pred)), 'home_team_win': pred})
pred_df.to_csv(f"{OUTPUT_DIR}/blending_kfold_optuna_{START_TIME}{DATA_SUFFIX}.csv", index=False)

print(f"execution time: {time.time() - start_time:.2f} sec")