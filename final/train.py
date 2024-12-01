import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time

RANDOM_STATE = 42

DATA_DIR = "_data"
OUTPUT_DIR = "output"

train_data = pd.read_csv(f"{DATA_DIR}/train_data.csv")
X_test = pd.read_csv(f"{DATA_DIR}/same_season_test_data.csv")

# Preprocess
print("start preprocessing...")
X = train_data.drop("home_team_win", axis=1).drop("date", axis=1)
y = train_data["home_team_win"]

CATEGORICAL_FEATURES = ['home_team_abbr', 'away_team_abbr', 'is_night_game', 'home_pitcher', 'away_pitcher', "home_team_season", "away_team_season"]
X = pd.get_dummies(X, columns=CATEGORICAL_FEATURES, dtype=float)
X.fillna(0, inplace=True)
y = y.map({True: 1, False: 0})
X_test = pd.get_dummies(X_test, columns=CATEGORICAL_FEATURES, dtype=float)
X_test.fillna(0, inplace=True)

for category_name in x.columns:
    if (category_name not in X_test.columns)
        


# scaler = StandardScaler()
# X = scaler.fit_transform(X)
# X_test = scaler.fit_transform(X_test)

# # Train the model
# print("start training...")
# train_start_time = time.time()
# model = SVC(kernel="rbf", gamma='auto', random_state=RANDOM_STATE)
# model.fit(X, y)

# print("\training time %ss" % (time.time() - train_start_time))

# # Make predictions on the test data
# print("start testing...")
# y_pred = model.predict(X_test)

# # Save the predictions
# print(y_pred);
# output = pd.DataFrame({"home_time_win": y_pred})
# output.to_csv(f"{OUTPUT_DIR}/predictions.csv", index=False)

# Save the model
# import pickle

# with open(f'{OUTPUT_DIR}/model.pkl','wb') as f:
#     pickle.dump(model, f)