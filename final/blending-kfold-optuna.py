# ref: https://www.kaggle.com/code/merrickolivier/blending-ensemble-for-classification/notebook
# ref: https://github.com/optuna/optuna-examples/blob/main/sklearn/sklearn_simple.py

import pandas as pd
import numpy as np

import os
import time
from datetime import datetime
import multiprocessing
from multiprocessing import Pool

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import optuna


RANDOM_STATE = None

DATA_DIR = "data/preprocess data"
OUTPUT_DIR = "predictions"
START_TIME = str(datetime.now().strftime("%m%d_%H%M"))
KFOLD = 5

BASE_MODELS = [
        ('etc', ExtraTreesClassifier()),
        # ('ada', AdaBoostClassifier()),
        ('dtc', DecisionTreeClassifier()),
        ('rf', RandomForestClassifier()),
        # ('gbc', GradientBoostingClassifier()),
    ]


def fit_kfold(model_name, model):
    
    kf = KFold(n_splits=KFOLD, shuffle=True, random_state=RANDOM_STATE)

    oof_pred = np.zeros((X.shape[0],))
    test_pred_list = np.zeros((X_test.shape[0], KFOLD))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"[{model_name}]\ttraining fold {fold+1}")
        start_time = time.time()
        
        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]
        
        model.fit(X_fold_train, y_fold_train)
        
        oof_pred[val_idx] = model.predict_proba(X_fold_val)[:, 1]
        test_pred_list[:, fold] = model.predict_proba(X_test)[:, 1]

        print(f"[{model_name}]\tfold {fold+1} took {time.time() - start_time:.2f} seconds")

    oof_pred = oof_pred.reshape(len(oof_pred), 1)
    auc_score = roc_auc_score(y, oof_pred)
    print(f'[{model_name}]\tAUC: {auc_score}')
    
    test_pred = test_pred_list.mean(axis=1)

    return oof_pred, test_pred, auc_score


def base_objective(trial, model_name):

    if model_name == 'etc':
        model = ExtraTreesClassifier(
            n_estimators=trial.suggest_int('n_estimators', 50, 500),
            max_depth=trial.suggest_int('max_depth', 3, 20),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
            random_state=RANDOM_STATE
        )

    elif model_name == 'ada':
        model = AdaBoostClassifier(
            n_estimators=trial.suggest_int('n_estimators', 50, 500),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 1),
            estimator=DecisionTreeClassifier(
                max_depth=trial.suggest_int('base_estimator_max_depth', 3, 20),
                min_samples_split=trial.suggest_int('base_estimator_min_samples_split', 2, 10)
                ),
            random_state=RANDOM_STATE
        )

    elif model_name == 'dtc':
        model = DecisionTreeClassifier(
            max_depth=trial.suggest_int('max_depth', 3, 20),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
            random_state=RANDOM_STATE
        )

    elif model_name == 'rf':
        model = RandomForestClassifier(
            n_estimators=trial.suggest_int('n_estimators', 50, 500),
            max_depth=trial.suggest_int('max_depth', 3, 20),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
            random_state=RANDOM_STATE
        )

    elif model_name == 'gbc':
        model = GradientBoostingClassifier(
            n_estimators=trial.suggest_int('n_estimators', 50, 500),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 1),
            max_depth=trial.suggest_int('max_depth', 3, 20),
            subsample=trial.suggest_discrete_uniform('subsample', 0.5, 1, 0.1),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
            random_state=RANDOM_STATE
        )

    _, _, auc_score = fit_kfold(model_name ,model)
    
    return auc_score


def optimize_base_model(model_name, base_model):    
    print(f"[{model_name}]\tOptimizing {model_name}...")
        
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: base_objective(trial, model_name), n_trials=20)
        
    print(f"[{model_name}]\tBest params for {model_name}: {study.best_params}")
        
    optimized_model = base_model.set_params(**study.best_params)
    
    return (model_name, optimized_model)


def optimize_base_models_pool():
    start_time = time.time()

    optimized_models = []
    
    cpus = multiprocessing.cpu_count()
    pool = Pool(cpus if cpus < len(BASE_MODELS) else len(BASE_MODELS))

    optimized_models = pool.starmap(optimize_base_model, BASE_MODELS)

    pool.close()

    print(f"Optimization done in {time.time() - start_time:.2f} seconds")
    print(optimized_models)

    return optimized_models


def train_meta_and_predict(optimized_models):

    preds_for_meta = []
    preds_for_test = []
    
    for name, model in optimized_models:
        print(f'training base model {name}')
        start_time = time.time()

        oof_pred, test_pred, _ = fit_kfold(name, model)
        preds_for_meta.append(oof_pred)
        preds_for_test.append(test_pred.reshape(-1, 1))
        
        print(f'{name} took {time.time() - start_time:.2f} seconds')

    print(f"training & testing meta model")
    start_time = time.time()
    meta_model = LogisticRegression()
    train_features = np.hstack(preds_for_meta)
    meta_model.fit(train_features, y)
    
    test_features = np.hstack(preds_for_test)
    meta_preds = meta_model.predict(test_features)
    print(f'Training & Testing done in {time.time() - start_time:.2f} seconds')

    meta_preds = meta_preds > 0.5 # convert to boolean

    return meta_preds


print(f"\n--- {START_TIME} ---\n")
root_start_time = time.time()

global X, y, X_test

X = pd.read_csv(f"{DATA_DIR}/train1_np.csv")
diff_X = pd.read_csv(f"{DATA_DIR}/train1_diff.csv")
y = pd.read_csv(f"{DATA_DIR}/train_y.csv")
X_test = pd.read_csv(f"{DATA_DIR}/test1_np.csv")
diff_X_test = pd.read_csv(f"{DATA_DIR}/test1_diff.csv")

X = pd.concat([X, diff_X], axis=1)
y = y.values.ravel()
X_test = pd.concat([X_test, diff_X_test], axis=1)

print(X)
print(y)
print(X_test)

optimized_models = optimize_base_models_pool()
pred = train_meta_and_predict(optimized_models)

pred_df = pd.DataFrame({'id': range(0, len(pred)), 'home_team_win': pred})
pred_df.to_csv(f"{OUTPUT_DIR}/blending_kfold_optuna_{START_TIME}.csv", index=False)

print(f"\n--- {time.time() - root_start_time:.2f} seconds ---\n")