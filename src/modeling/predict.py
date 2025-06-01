from sklearn.base import BaseEstimator


class FamilyModelSelector(BaseEstimator):
    def __init__(self, models_dict, family_col='family'):
        self.models_dict = models_dict
        self.family_col = family_col

    def fit(self, X, y=None):
        return self  # No fitting needed

    def predict(self, X):
        preds = []
        for _, row in X.iterrows():
            family = row[self.family_col]
            model = self.models_dict.get(family)
            if model is None:
                raise ValueError(f"No model found for family: {family}")
            row_input = row.drop(self.family_col).to_frame().T
            pred = model.predict(row_input)[0]
            preds.append(pred)
        return preds
