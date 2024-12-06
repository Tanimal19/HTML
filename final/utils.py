import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import csv

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

def one_hot_encoding(df):
    """
    One-hot encoding for categorical features
    """

    cat_df = df.select_dtypes(include=['object'])
    encode_cat_df = pd.get_dummies(cat_df)
    df = pd.concat([df, encode_cat_df], axis=1)
    df = df.drop(cat_df.columns, axis=1)

    return df


def standard_scaler(df):
    """
    Standard scaling for numerical features
    """

    num_df = df.select_dtypes(include=['int64', 'float64'])
    num_df = num_df.drop(['season'], axis=1) # season should not scaled
    scaled_num_df = StandardScaler().fit_transform(num_df)
    df[num_df.columns] = scaled_num_df

    return df


def align_columns(df1, df2):
    """
    Align columns for two dataframes
    """

    new_df1, new_df2 = df1.align(df2, join='outer', axis=1, fill_value=0)
    assert new_df1.columns.equals(new_df2.columns)

    return new_df1, new_df2


def select_important_features(X, y, threshold):
    """
    select important features by RandomForestClassifier,
    X should be preprocessed to fit RandomForestClassifier
    ref: https://gist.github.com/Keycatowo/eb042f1fdd5dd323e0a81a0670249bfb
    """

    feature_labels = list(X.columns)

    rf = RandomForestClassifier().fit(X, y)
    importances = rf.feature_importances_ 
    indices = np.argsort(importances)[::-1]

    # with open(f"output/preprocess/feature_importance.csv", "w") as f:
    #     writer = csv.writer(f)
        
    #     writer.writerow(["rank","feature","importance"])
    #     for i in range(X.shape[1]):
    #         writer.writerow([i + 1, feature_labels[indices[i]], importances[indices[i]]])

    selected_features = []
    for i in range(X.shape[1]):
        if importances[indices[i]] > threshold:
            selected_features.append(feature_labels[indices[i]])

    print(f"Selected features: {selected_features}")

    return selected_features

