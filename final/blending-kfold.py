# ref: https://www.kaggle.com/code/merrickolivier/blending-ensemble-for-classification/notebook

import pandas as pd
import numpy as np

import os
import time
from datetime import datetime

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


RANDOM_STATE = None

DATA_DIR = "data/preprocess data"
OUTPUT_DIR = "predictions"
START_TIME = str(datetime.now().strftime("%m%d_%H%M"))
KFOLD = 10


def fit_kfold(model, X, y, X_test):

    kf = KFold(n_splits=KFOLD, shuffle=True, random_state=RANDOM_STATE)

    oof_pred = np.zeros((X.shape[0],))
    test_pred_list = np.zeros((X_test.shape[0], KFOLD))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"training fold {fold+1}")
        start_time = time.time()
        
        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]
        
        model.fit(X_fold_train, y_fold_train)
        
        oof_pred[val_idx] = model.predict_proba(X_fold_val)[:, 1]
        test_pred_list[:, fold] = model.predict_proba(X_test)[:, 1]

        print(f"fold {fold+1} took {time.time() - start_time:.2f} seconds")

    oof_pred = oof_pred.reshape(len(oof_pred), 1)
    print(f'AUC: {roc_auc_score(y, oof_pred)}')
    
    test_pred = test_pred_list.mean(axis=1)

    return oof_pred, test_pred


def train_meta_and_predict(models, X, y, X_test):

    preds_for_meta = []
    preds_for_test = []
    
    for name, model in models:
        print(f'training {name}')
        start_time = time.time()

        oof_pred, test_pred = fit_kfold(model, X, y, X_test)
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
    print(f'meta took {time.time() - start_time:.2f} seconds')
    
    meta_preds = meta_preds > 0.5 # convert to boolean

    return meta_preds


MODELS = [
        ('etc', ExtraTreesClassifier()),
        ('ada', AdaBoostClassifier()),
        ('dtc', DecisionTreeClassifier()),
        ('rf', RandomForestClassifier()),
        ('gbc', GradientBoostingClassifier()),
        # ('svc', SVC(probability=True))
    ]


print(f"\n--- {START_TIME} ---\n")
root_start_time = time.time()

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

pred = train_meta_and_predict(MODELS, X, y, X_test)

pred_df = pd.DataFrame({'id': range(0, len(pred)), 'home_team_win': pred})
pred_df.to_csv(f"{OUTPUT_DIR}/blending_kfold_{START_TIME}.csv", index=False)

print(f"\n--- {time.time() - root_start_time:.2f} seconds ---\n")