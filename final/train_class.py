import abc
import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

import utils



class AbstractTrainMethod(metaclass=abc.ABCMeta):
    method_name = None
    random_state = None

    def run(self, raw_X, y, raw_X_test, prediction=True):
        print(f"[LOG]\tRunning {self.method_name}...")
        start_time = time.time()

        X, X_test = self.preprocess(raw_X, raw_X_test)
        print(f"[LOG]\tfinished preprocess in {time.time()-start_time:.2f} sec")

        model = self.train(X, y)
        print(f"[LOG]\tfinished training in {time.time()-start_time:.2f} sec")

        if not prediction:
            return model

        y_pred = model.predict(X_test)
        index = list(range(0,len(y_pred)))
        result = pd.DataFrame({"id": index, "home_team_win": y_pred})
        result["home_team_win"] = result["home_team_win"].map({1: True, 0: False})
        print(f"[LOG]\tfinished prediction in {time.time()-start_time:.2f} sec")

        return result

    @abc.abstractmethod
    def preprocess(self, raw_X, raw_X_test)->tuple:
        pass
    
    @abc.abstractmethod
    def train(self, X, y, *args, **kargs)->object:
        pass


class RandomForest(AbstractTrainMethod):
    method_name = 'RandomForest'
    random_state = None

    def preprocess(self, raw_X, raw_X_test):
        assert raw_X.isna().any().any() == False
        assert raw_X_test.isna().any().any() == False

        # one-hot encoding
        X = utils.one_hot_encoding(raw_X)
        X_test = utils.one_hot_encoding(raw_X_test)

        # align columns for X and X_test
        X_columns = set(X.columns)
        X_test_columns = set(X_test.columns)
        full_col = X_columns.union(X_test_columns)
        X_missing = pd.DataFrame(0, index=X.index, columns=list(full_col - X_columns))
        X = pd.concat([X, X_missing], axis=1)
        X_test_missing = pd.DataFrame(0, index=X_test.index, columns=list(full_col - X_test_columns))
        X_test = pd.concat([X_test, X_test_missing], axis=1)
        X = X[X_test.columns]

        assert X.isna().any().any() == False
        assert X_test.isna().any().any() == False
        
        return X, X_test

    def train(self, X, y, *args, **kwargs):
        decay_factor = kwargs.get("decay_factor", 5)
        weights = np.exp(-(2025 - X["season"]) / decay_factor)
        weights = weights / np.sum(weights)
        print(f"[INFO]\tweights: {weights}")

        model = RandomForestClassifier(random_state=self.random_state)
        model.fit(X, y, sample_weight=weights)

        return model
