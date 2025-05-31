import pandas as pd
from sklearn.base import TransformerMixin


class FeatureEngineer(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        pass
    
    def get_feature_names_out(self, input_features=None):
        pass
    
# Example usage
if __name__ == "__main__":
    # Create a sample DataFrame
    data = {
        'age': [25, 30, 35],
        'income': [50000, 60000, 70000],
    }
    df = pd.DataFrame(data)
    # Initialize the feature engineer
    feature_engineer = FeatureEngineer()
    # Fit and transform the DataFrame
    transformed_df = feature_engineer.fit_transform(df)