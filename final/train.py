import pandas as pd
import numpy as np
import time
from datetime import datetime
from train_class import RandomForest


TIME_STR = str(datetime.now().strftime("%m%d_%H%M"))
DATA_DIR = "_data"
OUTPUT_DIR = "output/predictions"


print(f"\n--- {TIME_STR} ---\n")

train_data = pd.read_csv(f"{DATA_DIR}/train_data_N.csv")
test_data = pd.read_csv(f"{DATA_DIR}/same_season_test_data_N.csv")
    
DROP_FEATURES = ["id", "home_team_season", "away_team_season"]
X = train_data.drop(["home_team_win", "date"] + DROP_FEATURES, axis=1)
y = train_data["home_team_win"].map({True: 1, False: 0})
X_test = test_data.drop(DROP_FEATURES, axis=1)

rf = RandomForest()
result = rf.run(X, y, X_test, prediction=True)

result.to_csv(f"{OUTPUT_DIR}/predictions_{TIME_STR}.csv", index=False)