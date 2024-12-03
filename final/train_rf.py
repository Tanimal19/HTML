from class_def import TrainMethod

class RandomForest(TrainMethod):
    method_name = 'RandomForest'

    def preprocess(self, raw_X, raw_y, raw_X_test):
        # numerical columns - fill with mean and standardize
        num_df = df[df.select_dtypes(include=['int64', 'float64']).columns]
        imputer = SimpleImputer(strategy='mean')
        num_df[num_df.columns] = imputer.fit_transform(num_df)

        # categorical columns - fill with "missing" and one-hot encoding
        cat_df = df[df.select_dtypes(include=['object']).columns]
        cat_df = cat_df.fillna("missing")
        cat_df = pd.get_dummies(cat_df, dtype=float)

        # boolean columns - randomly select 0 or 1
        bool_df = df[df.select_dtypes(include=['bool']).columns]
        for col in bool_df.columns:
            bool_df[col] = bool_df[col].map({True: 1, False: 0})
            bool_df[col] = bool_df[col].map(lambda x: x if pd.notnull(x) else bool(pd.np.random.choice([0, 1])))
        
        return X, y, X_test

    def train(self, X, y):
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=0)
        model.fit(X, y)
        return model

    def predict(self, model, X_test):
        return model.predict(X_test)