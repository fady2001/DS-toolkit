from datetime import timedelta
import warnings

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from src.config import RAW_DATA_DIR

warnings.filterwarnings("ignore")


class TemporalFeatureTransformer(BaseEstimator, TransformerMixin):
    """Create temporal features from date column"""

    def __init__(self, date_column="date"):
        self.date_column = date_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        def days_since_payday(date):
            day = date.day
            if day <= 15:
                # Days since last month's end
                last_month_end = date.replace(day=1) - timedelta(days=1)
                return (date - last_month_end).days
            else:
                # Days since 15th of current month
                current_month_15th = date.replace(day=15)
                return (date - current_month_15th).days

        def days_until_payday(date):
            day = date.day
            if day < 15:
                # Days until 15th
                return 15 - day
            else:
                # Days until month end
                next_month = date.replace(day=28) + timedelta(days=4)
                month_end = next_month - timedelta(days=next_month.day)
                return (month_end - date).days

        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(X[self.date_column]):
            X[self.date_column] = pd.to_datetime(X[self.date_column])

        # Create temporal features
        X["year"] = X[self.date_column].dt.year
        X["month"] = X[self.date_column].dt.month
        X["day"] = X[self.date_column].dt.day
        X["dayofweek"] = X[self.date_column].dt.dayofweek
        X["weekofyear"] = X[self.date_column].dt.isocalendar().week
        X["day_of_year"] = X[self.date_column].dt.dayofyear
        X["is_weekend"] = (X["dayofweek"] >= 5).astype(int)
        X["is_month_start"] = X[self.date_column].dt.is_month_start.astype(int)
        X["is_month_end"] = X[self.date_column].dt.is_month_end.astype(int)
        X["is_quarter_start"] = X[self.date_column].dt.is_quarter_start.astype(int)
        X["is_quarter_end"] = X[self.date_column].dt.is_quarter_end.astype(int)
        X["is_payday"] = ((X["day"] == 15) | X[self.date_column].dt.is_month_end).astype(int)
        X["days_since_payday"] = X[self.date_column].apply(days_since_payday)
        X["days_until_payday"] = X[self.date_column].apply(days_until_payday)

        return X
    
    def fit_transform(self, X):
        """Fit and transform the data"""
        return self.fit(X).transform(X)


class HolidayFeatureTransformer(BaseEstimator, TransformerMixin):
    """Create holiday features from holidays dataset"""

    def __init__(self, holiday_df_path=f"{RAW_DATA_DIR}/holidays_events.csv", date_column="date"):
        self.holiday_df_path = holiday_df_path
        self.date_column = date_column
        self.holiday_sets = {}

    def fit(self, X, y=None):
        # Load and process holidays data
        holiday_df = pd.read_csv(self.holiday_df_path, parse_dates=["date"])

        self.holiday_sets = {
            "national_holidays": holiday_df[holiday_df["locale"] == "National"]["date"].unique(),
            "regional_holidays": holiday_df[holiday_df["locale"] == "Regional"]["date"].unique(),
            "local_holidays": holiday_df[holiday_df["locale"] == "Local"]["date"].unique(),
            "additional_holidays": holiday_df[holiday_df["type"] == "Additional"]["date"].unique(),
            "working_days": holiday_df[holiday_df["type"] == "Work Day"]["date"].unique(),
            "events": holiday_df[holiday_df["type"] == "Event"]["date"].unique(),
            "bridge_days": holiday_df[holiday_df["type"] == "Bridge"]["date"].unique(),
            "transferred_days": holiday_df[holiday_df["transferred"] is True]["date"].unique(),
        }
        return self

    def transform(self, X):
        X = X.copy()

        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(X[self.date_column]):
            X[self.date_column] = pd.to_datetime(X[self.date_column])

        # Create holiday features
        X["is_national_holiday"] = (
            X[self.date_column].isin(self.holiday_sets["national_holidays"]).astype(int)
        )
        X["is_regional_holiday"] = (
            X[self.date_column].isin(self.holiday_sets["regional_holidays"]).astype(int)
        )
        X["is_local_holiday"] = (
            X[self.date_column].isin(self.holiday_sets["local_holidays"]).astype(int)
        )
        X["is_additional_holiday"] = (
            X[self.date_column].isin(self.holiday_sets["additional_holidays"]).astype(int)
        )
        X["is_working_day"] = (
            X[self.date_column].isin(self.holiday_sets["working_days"]).astype(int)
        )
        X["is_event"] = X[self.date_column].isin(self.holiday_sets["events"]).astype(int)
        X["is_bridge_day"] = X[self.date_column].isin(self.holiday_sets["bridge_days"]).astype(int)
        X["is_transferred_day"] = (
            X[self.date_column].isin(self.holiday_sets["transferred_days"]).astype(int)
        )

        return X


class LagFeatureTransformer(BaseEstimator, TransformerMixin):
    """Create lag features for time series"""

    def __init__(
        self, target_col="sales", lags=[1, 7, 14, 30], group_cols=["store_nbr", "family"]
    ):
        self.target_col = target_col
        self.lags = lags
        self.group_cols = group_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X_sorted = X.sort_values(self.group_cols + ["date"])

        for lag in self.lags:
            X_sorted[f"{self.target_col}_lag_{lag}"] = X_sorted.groupby(self.group_cols)[
                self.target_col
            ].shift(lag)

        return X_sorted


class RollingFeatureTransformer(BaseEstimator, TransformerMixin):
    """Create rolling window statistics"""

    def __init__(
        self, target_col="sales", windows=[7, 14, 30], group_cols=["store_nbr", "family"]
    ):
        self.target_col = target_col
        self.windows = windows
        self.group_cols = group_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X_sorted = X.sort_values(self.group_cols + ["date"])

        for window in self.windows:
            # Rolling mean
            X_sorted[f"{self.target_col}_rolling_mean_{window}"] = X_sorted.groupby(
                self.group_cols
            )[self.target_col].transform(lambda x: x.rolling(window=window, min_periods=1).mean())

            # Rolling std
            X_sorted[f"{self.target_col}_rolling_std_{window}"] = X_sorted.groupby(
                self.group_cols
            )[self.target_col].transform(lambda x: x.rolling(window=window, min_periods=1).std())

            # Rolling max
            X_sorted[f"{self.target_col}_rolling_max_{window}"] = X_sorted.groupby(
                self.group_cols
            )[self.target_col].transform(lambda x: x.rolling(window=window, min_periods=1).max())

            # Rolling min
            X_sorted[f"{self.target_col}_rolling_min_{window}"] = X_sorted.groupby(
                self.group_cols
            )[self.target_col].transform(lambda x: x.rolling(window=window, min_periods=1).min())

        return X_sorted

#TODO: call function to handle missing date indices in external data
class ExternalDataMerger(BaseEstimator, TransformerMixin):
    """Merge external datasets (oil, stores, transactions)"""

    def __init__(
        self,
        oil_df_path="../data/raw/oil.csv",
        stores_df_path="../data/raw/stores.csv",
        transactions_df_path="../data/raw/transactions.csv",
    ):
        self.oil_df_path = oil_df_path
        self.stores_df_path = stores_df_path
        self.transactions_df_path = transactions_df_path
        self.oil_df = None
        self.stores_df = None
        self.transactions_df = None

    def fit(self, X):
        # Load and preprocess external data
        self.oil_df = pd.read_csv(self.oil_df_path, parse_dates=["date"])
        self.stores_df = pd.read_csv(self.stores_df_path)
        self.transactions_df = pd.read_csv(self.transactions_df_path, parse_dates=["date"])

        # Interpolate missing oil prices
        all_dates = pd.date_range(start=self.oil_df["date"].min(), end=self.oil_df["date"].max())
        self.oil_df = (
            self.oil_df.set_index("date").reindex(all_dates).rename_axis("date").reset_index()
        )
        self.oil_df["dcoilwtico"] = self.oil_df["dcoilwtico"].interpolate(
            method="polynomial", order=2
        )
        self.oil_df["dcoilwtico"] = self.oil_df["dcoilwtico"].bfill()

        return self

    def transform(self, X):
        X = X.copy()

        # Merge oil data
        X = X.merge(self.oil_df, on="date", how="left")

        # Merge stores data
        X = X.merge(self.stores_df, on="store_nbr", how="left")

        # Merge transactions data
        X = X.merge(self.transactions_df, on=["date", "store_nbr"], how="left")

        # Fill missing transactions with 0
        X["transactions"] = X["transactions"].fillna(0)

        return X



class TestDataLagTransformer(BaseEstimator, TransformerMixin):
    """Create lag features for test data using training data to avoid leakage"""

    def __init__(
        self, target_col="sales", lags=[1, 7, 14, 30], group_cols=["store_nbr", "family"]
    ):
        self.target_col = target_col
        self.lags = lags
        self.group_cols = group_cols
        self.train_df = None

    def fit(self, X, y=None):
        # Store training data for creating test features
        self.train_df = X.copy()
        return self

    def transform(self, X):
        # Combine train and test for consistent processing
        combined_df = pd.concat([self.train_df, X], ignore_index=True)
        combined_df = combined_df.sort_values(self.group_cols + ["date"])

        # Create lag features
        for lag in self.lags:
            combined_df[f"{self.target_col}_lag_{lag}"] = combined_df.groupby(self.group_cols)[
                self.target_col
            ].shift(lag)

        # Return only test data portion
        test_start_date = X["date"].min()
        test_with_lags = combined_df[combined_df["date"] >= test_start_date].copy()

        return test_with_lags


class TestDataRollingTransformer(BaseEstimator, TransformerMixin):
    """Create rolling features for test data using training data"""

    def __init__(
        self, target_col="sales", windows=[7, 14, 30], group_cols=["store_nbr", "family"]
    ):
        self.target_col = target_col
        self.windows = windows
        self.group_cols = group_cols
        self.train_df = None

    def fit(self, X, y=None):
        self.train_df = X.copy()
        return self

    def transform(self, X):
        # Combine train and test data
        combined_df = pd.concat([self.train_df, X], ignore_index=True)
        combined_df = combined_df.sort_values(self.group_cols + ["date"])

        for window in self.windows:
            # Rolling mean
            combined_df[f"{self.target_col}_rolling_mean_{window}"] = combined_df.groupby(
                self.group_cols
            )[self.target_col].transform(lambda x: x.rolling(window=window, min_periods=1).mean())

            # Rolling std
            combined_df[f"{self.target_col}_rolling_std_{window}"] = combined_df.groupby(
                self.group_cols
            )[self.target_col].transform(lambda x: x.rolling(window=window, min_periods=1).std())

            # Rolling max
            combined_df[f"{self.target_col}_rolling_max_{window}"] = combined_df.groupby(
                self.group_cols
            )[self.target_col].transform(lambda x: x.rolling(window=window, min_periods=1).max())

            # Rolling min
            combined_df[f"{self.target_col}_rolling_min_{window}"] = combined_df.groupby(
                self.group_cols
            )[self.target_col].transform(lambda x: x.rolling(window=window, min_periods=1).min())

        # Return only test data portion
        test_start_date = X["date"].min()
        test_with_rolling = combined_df[combined_df["date"] >= test_start_date].copy()

        return test_with_rolling


class NullHandlerTransformer(BaseEstimator, TransformerMixin):
    """Handle null values for prediction"""

    def __init__(self, strategy="comprehensive"):
        self.strategy = strategy

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if self.strategy == "comprehensive":
            # Forward fill lag features within groups
            lag_cols = [col for col in X.columns if "lag" in col]
            for col in lag_cols:
                X[col] = X.groupby(["store_nbr", "family"])[col].fillna(method="ffill")

            # Forward fill rolling features within groups
            rolling_cols = [col for col in X.columns if "rolling" in col]
            for col in rolling_cols:
                X[col] = X.groupby(["store_nbr", "family"])[col].fillna(method="ffill")

            # For remaining nulls in lag features, use group median then 0
            for col in lag_cols:
                group_medians = X.groupby(["store_nbr", "family"])[col].transform("median")
                X[col] = X[col].fillna(group_medians).fillna(0)

            # For remaining nulls in rolling features
            for col in rolling_cols:
                if "std" in col:
                    X[col] = X[col].fillna(0)
                else:
                    group_medians = X.groupby(["store_nbr", "family"])[col].transform("median")
                    X[col] = X[col].fillna(group_medians).fillna(0)

        return X


def create_training_pipeline():
    """Create complete feature engineering pipeline for training data"""

    pipeline = Pipeline(
        [
            ("temporal_features", TemporalFeatureTransformer()),
            ("holiday_features", HolidayFeatureTransformer()),
            ("external_data", ExternalDataMerger()),
            ("rolling_features", RollingFeatureTransformer()),
            ("lag_features", LagFeatureTransformer()),
            ("null_handler", NullHandlerTransformer()),
        ]
    )

    return pipeline


def create_test_pipeline(train_df):
    """Create feature engineering pipeline for test data to avoid leakage"""

    # First, create features that don't depend on target variable
    base_pipeline = Pipeline(
        [
            ("temporal_features", TemporalFeatureTransformer()),
            ("holiday_features", HolidayFeatureTransformer()),
            ("external_data", ExternalDataMerger()),
        ]
    )

    # Then create target-dependent features using training data
    target_pipeline = Pipeline(
        [
            ("test_rolling_features", TestDataRollingTransformer()),
            ("test_lag_features", TestDataLagTransformer()),
            ("null_handler", NullHandlerTransformer()),
        ]
    )

    return base_pipeline, target_pipeline


def get_feature_columns():
    """Get list of feature columns for model training"""

    # Base features
    temporal_features = [
        "year",
        "month",
        "day",
        "dayofweek",
        "weekofyear",
        "day_of_year",
        "is_weekend",
        "is_month_start",
        "is_month_end",
        "is_quarter_start",
        "is_quarter_end",
        "is_payday",
        "days_since_payday",
        "days_until_payday",
    ]

    holiday_features = [
        "is_national_holiday",
        "is_regional_holiday",
        "is_local_holiday",
        "is_additional_holiday",
        "is_working_day",
        "is_event",
        "is_bridge_day",
        "is_transferred_day",
    ]

    external_features = ["dcoilwtico", "type", "cluster", "transactions"]

    # Lag features
    lag_features = [f"sales_lag_{lag}" for lag in [1, 7, 14, 30]]

    # Rolling features
    rolling_features = []
    for window in [7, 14, 30]:
        for stat in ["mean", "std", "max", "min"]:
            rolling_features.append(f"sales_rolling_{stat}_{window}")

    # Categorical features to encode
    categorical_features = ["family", "store_nbr", "type", "state", "city"]

    feature_columns = {
        "temporal": temporal_features,
        "holiday": holiday_features,
        "external": external_features,
        "lag": lag_features,
        "rolling": rolling_features,
        "categorical": categorical_features,
    }

    return feature_columns


if __name__ == "__main__":
    # Example usage
    print("Feature Engineering Pipeline Created Successfully!")
    print("\nAvailable components:")
    print("1. TemporalFeatureTransformer - Creates time-based features")
    print("2. HolidayFeatureTransformer - Creates holiday features")
    print("3. LagFeatureTransformer - Creates lag features")
    print("4. RollingFeatureTransformer - Creates rolling statistics")
    print("5. ExternalDataMerger - Merges oil, stores, transactions data")
    print("6. NullHandlerTransformer - Handles missing values")
    print("7. TestDataLagTransformer - Safe lag features for test data")
    print("8. TestDataRollingTransformer - Safe rolling features for test data")

    print("\nPipelines:")
    print("- create_training_pipeline() - Complete training pipeline")
    print("- create_test_pipeline() - Test pipeline avoiding data leakage")

    # Show feature categories
    features = get_feature_columns()
    print(f"\nFeature categories:")
    for category, cols in features.items():
        print(f"{category}: {len(cols)} features")
        
    # apply the training pipeline to a sample DataFrame
    train_df = pd.read_csv("../data/raw/train.csv", parse_dates=["date"])
    train_pipeline = create_training_pipeline()
    transformed_train_df = train_pipeline.fit_transform(train_df)
    print("\nTransformed Training DataFrame:")
    print(transformed_train_df.head())