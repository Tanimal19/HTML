import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
import csv


DATA_DIR = "data/preprocess data"
OUTPUT_DIR = "data/preprocess data"


# find features with high correlation
if False:
    X = pd.read_csv(f"{DATA_DIR}/train1.csv")

    num_X = X.select_dtypes(include=['int64', 'float64'])
    corr_map = num_X.corr()
    threshold = 0.8
    with open(f"{OUTPUT_DIR}/high_correlation_features.txt", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Feature 1", "Feature 2", "Correlation"])
        for i in range(len(corr_map.columns)):
            for j in range(i):
                if abs(corr_map.iloc[i, j]) > threshold:
                    writer.writerow([corr_map.columns[i], corr_map.columns[j], corr_map.iloc[i, j]])


# compare distribution of features between train and test data
if False:
    X = pd.read_csv(f"{DATA_DIR}/train1.csv")
    X_test = pd.read_csv(f"{DATA_DIR}/test1.csv")

    num_X = X.select_dtypes(include=['int64', 'float64'])
    nmu_X_test = X_test.select_dtypes(include=['int64', 'float64'])

    num_features = len(num_X.columns)
    cols = 4
    rows = (num_features + cols - 1) // cols

    plt.figure(figsize=(24, 6*rows))
    for i, feature in enumerate(list(num_X.columns)):
        print(f"[{i}] plotting {feature}")
        plt.subplot(rows, cols, i+1)
        sns.histplot(num_X[feature], kde=True)
        sns.histplot(nmu_X_test[feature], kde=True, color='green')
    plt.tight_layout()
    plt.show()
    plt.savefig(f"{OUTPUT_DIR}/X_distribution.png")
