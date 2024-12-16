import csv
import pandas as pd

df = pd.read_csv("predictions/blending_kfold_optuna_1211_0939.csv")

pred = df['home_team_win'].values
pred_bool = pred > 0.5
df['home_team_win'] = pred_bool.astype(bool)

df.to_csv("predictions/blending_kfold_optuna_1211_0939.csv", index=False)