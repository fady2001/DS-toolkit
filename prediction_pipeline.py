import pickle

import numpy as np
import pandas as pd

from src.config import MODELS_DIR, RAW_DATA_DIR, SUBMISSIONS_DIR
from src.dataset import scrap_directory


def load_trained_pipeline(holidays_df, oil_df, transactions_df, stores_df):
    """Load the trained pipeline from disk."""
    pipeline_path = MODELS_DIR / "full_pipeline.pkl"

    if not pipeline_path.exists():
        raise FileNotFoundError(
            f"Trained pipeline not found at {pipeline_path}. Please run the training pipeline first."
        )

    with open(pipeline_path, "rb") as f:
        pipeline = pickle.load(f)
        
    pipeline.named_steps["feature_engineering"].named_steps["lag_features"].set_mode(is_train=True)
    pipeline.named_steps["feature_engineering"].named_steps["rolling_features"].set_mode(is_train=False)
    pipeline.named_steps["feature_engineering"].named_steps["holiday_features"].set_holiday_sets(
        holidays_df if holidays_df is not None else pd.read_csv(f"{RAW_DATA_DIR}/holidays.csv")
    )
    pipeline.named_steps["feature_engineering"].named_steps["external_data"].set_mode(is_train=False)
    pipeline.named_steps["feature_engineering"].named_steps["external_data"].set_dataframes(
        oil_df if oil_df is not None else pd.read_csv(f"{RAW_DATA_DIR}/oil.csv"),
        stores_df if stores_df is not None else pd.read_csv(f"{RAW_DATA_DIR}/stores.csv"),
        transactions_df if transactions_df is not None else pd.read_csv(f"{RAW_DATA_DIR}/transactions.csv"),
    )
    pipeline.named_steps["feature_engineering"].named_steps["null_handler"].set_strategy("comprehensive")
    print(f"Pipeline loaded successfully from {pipeline_path}")
    return pipeline


def prepare_test_data():
    """Load and prepare test data for prediction."""
    # Load all datasets (needed for feature engineering)
    datasets = scrap_directory(RAW_DATA_DIR, file_extension=".csv")

    # Parse date columns in datasets
    for df_name, df in datasets.items():
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            datasets[df_name] = df

    test_df = datasets.get("test", None)
    test_df["sales"] = np.nan # Initialize sales column for test set
    holidays_df = datasets.get("holidays_events", None)
    oil_df = datasets.get("oil", None)
    transactions_df = datasets.get("transactions", None)
    stores_df = datasets.get("stores", None)
    if test_df is None:
        raise FileNotFoundError("Test dataset not found in the raw data directory.")

    print(f"Test data loaded with shape: {test_df.shape}")
    return test_df, holidays_df, oil_df, transactions_df, stores_df



def create_submission_file(test_df, predictions, output_path=None):
    """Create a submission file with ID and predictions."""
    if output_path is None:
        output_path = SUBMISSIONS_DIR / "test_predictions.csv"

    # Create submission DataFrame
    submission_df = pd.DataFrame({"id": test_df["id"], "sales": predictions})

    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    print(f"Submission file shape: {submission_df.shape}")
    print(f"Sample predictions:\n{submission_df.head()}")

    return submission_df


def main():
    """Main function to run the prediction pipeline."""
    try:
        print("=" * 50)
        print("STORE SALES PREDICTION PIPELINE")
        print("=" * 50)

        # Step 1: Prepare test data
        print("\n2. Loading and preparing test data...")
        test_df,holidays_df, oil_df, transactions_df, stores_df = prepare_test_data()
        
        # Step 2: Load the trained pipeline
        print("\n1. Loading trained pipeline...")
        pipeline = load_trained_pipeline(holidays_df, oil_df, transactions_df, stores_df)


        # Step 3: Make predictions
        print("\n3. Making predictions...")
        transformed = pipeline.named_steps["feature_engineering"].transform(test_df)
        original_family = transformed["family"].copy()
        
        processed = pipeline.named_steps["preprocessing"].transform(transformed)
        
        final_test = pd.DataFrame(processed, columns=transformed.columns.drop(["id", "date"]))
        final_test.rename(columns={"family": "family_encoded"}, inplace=True)
        final_test["family"] = original_family.values
                
        predictions = pipeline.named_steps["model_selector"].predict(final_test)
        
        # Step 4: Create submission file
        print("\n4. Creating submission file...")
        submission_df = create_submission_file(test_df, predictions)

        print("\n" + "=" * 50)
        print("PREDICTION PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 50)

        # Print some statistics
        print("\nPrediction Statistics:")
        print(f"Total predictions: {len(predictions)}")
        print(f"Mean prediction: {submission_df['sales'].mean():.2f}")
        print(f"Min prediction: {submission_df['sales'].min():.2f}")
        print(f"Max prediction: {submission_df['sales'].max():.2f}")
        print(f"Std prediction: {submission_df['sales'].std():.2f}")

        return submission_df

    except Exception as e:
        print(f"Error in prediction pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    submission_df = main()
