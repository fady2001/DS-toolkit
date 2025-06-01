import pandas as pd

from src.config import INTERIM_DATA_DIR, RAW_DATA_DIR
from src.dataset import scrap_directory
from src.feature_pipeline import create_features_pipeline

if __name__ == "__main__":
    # load datasets    
    datasets = scrap_directory(RAW_DATA_DIR, file_extension=".csv")
    
    # parse date columns in datasets
    for df_name, df in datasets.items():
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            datasets[df_name] = df
            
    # create feature engineering pipeline
    feature_pipeline = create_features_pipeline(
        holiday_df= datasets.get("holidays_events", None),
        oil_df= datasets.get("oil", None),
        transactions_df= datasets.get("transactions", None),
        stores_df= datasets.get("stores", None),
    )
    
    
    # apply feature engineering pipeline to train dataset
    train_df:pd.DataFrame = feature_pipeline.fit_transform(datasets.get("train", None))
    
    # save the transformed train dataset as parquet file
    train_df.to_parquet(INTERIM_DATA_DIR / "train_transformed.parquet",engine="pyarrow", index=False)
    