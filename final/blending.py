# ref: https://www.kaggle.com/code/merrickolivier/blending-ensemble-for-classification/notebook

import pandas as pd
import numpy as np

import os
import time
from datetime import datetime

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


RANDOM_STATE = None

DATA_DIR = "preprocess data"
OUTPUT_DIR = "predictions"
START_TIME = str(datetime.now().strftime("%m%d_%H%M"))


print(f"\n--- {START_TIME} ---\n")
root_start_time = time.time()

X = pd.read_csv(f"{DATA_DIR}/train1.csv")
y = pd.read_csv(f"{DATA_DIR}/train_y.csv")
y = y.values.ravel()
X_test = pd.read_csv(f"{DATA_DIR}/test1.csv")

print(X)
print(y)
print(X_test)

def fit_models(models, X_train, X_val, y_train, y_val):

    preds_for_meta = []
    
    for name, model in models:        
        model.fit(X_train, y_train)
        pred = model.predict_proba(X_val)[:, 1]
        
        roc_base = roc_auc_score(y_val, pred)
        print(f'{name} AUC: {roc_base}')
        
        pred = pred.reshape(len(pred), 1)        
        preds_for_meta.append(pred)
        
    meta_features = np.hstack(preds_for_meta)
    
    meta_model = LogisticRegression()
    meta_model.fit(meta_features, y_val)
    
    print(f'meta AUC: {roc_auc_score(y_val, meta_model.predict_proba(meta_features)[:, 1])}')
    
    return meta_model


def meta_predict(models, meta_model, X_test):
    
    preds_for_meta = []
    
    for name, model in models:
        pred = model.predict(X_test)        
        pred = pred.reshape(len(pred), 1)        
        preds_for_meta.append(pred)
        
    meta_features = np.hstack(preds_for_meta)
    
    meta_preds = meta_model.predict(meta_features)
    
    return meta_preds


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

MODELS = [('etc', ExtraTreesClassifier()),
        ('ada', AdaBoostClassifier()),
        ('dtc', DecisionTreeClassifier()),
        ('rf', RandomForestClassifier()),
        ('gbc', GradientBoostingClassifier())
        ('svc', SVC(probability=True))]

meta_model = fit_models(MODELS, X_train, X_val, y_train, y_val)
pred = meta_predict(MODELS, meta_model, X_test)

pred_df = pd.DataFrame({'id': range(0, len(pred)), 'home_team_win': pred})
pred_df.to_csv(f"{OUTPUT_DIR}/blending_{START_TIME}.csv", index=False)