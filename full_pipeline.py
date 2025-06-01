import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import INTERIM_DATA_DIR, MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.dataset import scrap_directory
from src.feature_pipeline import create_features_pipeline
from src.modeling.predict import FamilyModelSelector
from src.modeling.train import train_model_for_each_family
from src.Preprocessor import Preprocessor

if __name__ == "__main__":
    ######################################## 1. Load datasets ########################################
    datasets = scrap_directory(RAW_DATA_DIR, file_extension=".csv")
    
    # parse date columns in datasets
    for df_name, df in datasets.items():
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            datasets[df_name] = df
            
    ######################################## 2. Create feature engineering pipeline ########################################
    feature_pipeline = create_features_pipeline(
        holiday_df= datasets.get("holidays_events", None),
        oil_df= datasets.get("oil", None),
        transactions_df= datasets.get("transactions", None),
        stores_df= datasets.get("stores", None),
    )
    
    
    # apply feature engineering pipeline to train dataset
    feature_engineered_train_df:pd.DataFrame = feature_pipeline.fit_transform(datasets.get("train", None))
    
    # save the transformed train dataset as parquet file
    feature_engineered_train_df.to_parquet(INTERIM_DATA_DIR / "feature_engineered_train_df.parquet",engine="pyarrow", index=False)
    
    ########################################## 3. preprocessing ##########################################################
    feature_engineered_train_df = pd.read_parquet(INTERIM_DATA_DIR / "feature_engineered_train_df.parquet", engine="pyarrow")
    categorical_columns = [col for col in feature_engineered_train_df.columns if feature_engineered_train_df[col].dtype == "object"]

    config = {
        "drop": ["id", "date"],
        "encoding": {
            "ordinal": categorical_columns,
        },
    }

    preprocessor = Preprocessor(pipeline_config=config)
    preprocessor.fit(feature_engineered_train_df)
    transformed_data = preprocessor.transform(feature_engineered_train_df)
    print("transformed_data shape:", transformed_data.shape)
    # Get feature names from the preprocessor
    feature_names = preprocessor.get_feature_names_from_preprocessor()
    print(f"Feature names after encoding: {feature_names}")
    # Convert numpy array back to DataFrame
    transformed_df = pd.DataFrame(transformed_data, columns=feature_engineered_train_df.columns.drop(["id", "date"]))

    # rename column family to family_encoded
    transformed_df.rename(columns={"family": "family_encoded"}, inplace=True)
    transformed_df["family"] = feature_engineered_train_df["family"]
    transformed_df.to_parquet(PROCESSED_DATA_DIR / "processed_train_df.parquet", engine="pyarrow", index=False)
    
    ########################################## 4. training ##########################################################
    processed_train_df = pd.read_parquet(PROCESSED_DATA_DIR / "processed_train_df.parquet", engine="pyarrow")
    models = train_model_for_each_family(processed_train_df)
    
    model_selector = FamilyModelSelector(models_dict=models)
    
    full_pipeline = Pipeline(steps=[
        ("feature_engineering", feature_pipeline),
        ("preprocessing", preprocessor),
        ("model_selector", model_selector)
    ])
    
    # Save the full pipeline to disk
    import pickle
    
    with open(MODELS_DIR / "full_pipeline.pkl", "wb") as f:
        pickle.dump(full_pipeline, f)
    print("Full pipeline saved to disk.")