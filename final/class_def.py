import abc
import pandas as pd

class TrainMethod(metaclass=abc.ABCMeta):
    method_name = None

    @abc.abstractmethod
    def preprocess(self, raw_X, raw_y, raw_X_test)->tuple:
        pass
    
    @abc.abstractmethod
    def train(self, X, y)->object:
        pass

    def predict(self, model, X_test)->pd.DataFrame:
        pass
