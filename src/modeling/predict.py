import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator


class FamilyModelSelector(BaseEstimator):
    def __init__(self, models_dict, family_col="family"):
        self.models_dict = models_dict
        self.family_col = family_col

    def fit(self, X, y=None):
        return self  # No fitting needed

    def predict(self, X):
        # Create a copy to avoid modifying the original dataframe
        X_copy = X.copy()

        # Initialize predictions array with NaN values
        predictions = np.full(len(X_copy), np.nan)

        # Group by family and process each group
        for family, group_indices in X_copy.groupby(self.family_col).groups.items():
            model = self.models_dict.get(family)
            if model is None:
                raise ValueError(f"No model found for family: {family}")

            # Get the subset of data for this family
            family_data = X_copy.loc[group_indices]

            # Prepare features by dropping family column and sales column if present
            features_to_drop = [self.family_col]
            if "sales" in family_data.columns:
                features_to_drop.append("sales")

            family_features = family_data.drop(columns=features_to_drop)
            family_features = family_features.apply(pd.to_numeric, errors="coerce")

            # Make batch prediction for this family
            family_predictions = model.predict(family_features)

            # Store predictions in the correct positions
            predictions[group_indices] = family_predictions

        return predictions
