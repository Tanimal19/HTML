import pandas as pd


def one_hot_encoding(df):
    cat_df = df.select_dtypes(include=['object'])
    encode_cat_df = pd.get_dummies(cat_df)
    df = pd.concat([df, encode_cat_df], axis=1)
    df = df.drop(cat_df.columns, axis=1)
    return df