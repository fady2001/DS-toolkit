from datetime import datetime
import logging
import pickle
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
import uvicorn

from src.config import MODELS_DIR, RAW_DATA_DIR
from src.dataset import scrap_directory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Store Sales Prediction API",
    description="API for predicting store sales using a trained machine learning pipeline",
    version="1.0.0",
)

# Global variables to store loaded pipeline and reference data
pipeline = None
holidays_df = None
oil_df = None
transactions_df = None
stores_df = None


class PredictionRequest(BaseModel):
    """Single prediction request model"""

    id: int = Field(..., description="Unique identifier for the prediction")
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    store_nbr: int = Field(..., description="Store number")
    family: str = Field(..., description="Product family")
    onpromotion: Optional[int] = Field(0, description="Number of items on promotion")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request model"""

    predictions: List[PredictionRequest] = Field(..., description="List of prediction requests")


class PredictionResponse(BaseModel):
    """Single prediction response model"""

    id: int
    predicted_sales: float
    status: str = "success"


class BatchPredictionResponse(BaseModel):
    """Batch prediction response model"""

    predictions: List[PredictionResponse]
    total_predictions: int
    status: str = "success"


def load_pipeline_and_data():
    """Load the trained pipeline and reference data"""
    global pipeline, holidays_df, oil_df, transactions_df, stores_df

    try:
        # Load the trained pipeline - using the actual model file from your workspace
        pipeline_path = MODELS_DIR / "full_pipeline.pkl"

        if not pipeline_path.exists():
            raise FileNotFoundError(f"Trained pipeline not found at {pipeline_path}")

        with open(pipeline_path, "rb") as f:
            pipeline = pickle.load(f)

        # Load reference datasets needed for feature engineering
        datasets = scrap_directory(RAW_DATA_DIR, file_extension=".csv")

        # Parse date columns in datasets
        for df_name, df in datasets.items():
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                datasets[df_name] = df

        holidays_df = datasets.get("holidays_events")
        oil_df = datasets.get("oil")
        transactions_df = datasets.get("transactions")
        stores_df = datasets.get("stores")

        # Configure pipeline for inference mode if it has these components
        try:
            if hasattr(pipeline, "named_steps"):
                if "feature_engineering" in pipeline.named_steps:
                    fe_step = pipeline.named_steps["feature_engineering"]
                    if hasattr(fe_step, "named_steps"):
                        if "lag_features" in fe_step.named_steps:
                            fe_step.named_steps["lag_features"].set_mode(is_train=True)
                        if "rolling_features" in fe_step.named_steps:
                            fe_step.named_steps["rolling_features"].set_mode(is_train=False)
                        if "holiday_features" in fe_step.named_steps:
                            fe_step.named_steps["holiday_features"].set_holiday_sets(holidays_df)
                        if "external_data" in fe_step.named_steps:
                            fe_step.named_steps["external_data"].set_mode(is_train=False)
                            fe_step.named_steps["external_data"].set_dataframes(
                                oil_df, stores_df, transactions_df
                            )
                        if "null_handler" in fe_step.named_steps:
                            fe_step.named_steps["null_handler"].set_strategy("comprehensive")
        except Exception as config_error:
            logger.warning(f"Could not configure pipeline components: {config_error}")

        logger.info("Pipeline and reference data loaded successfully")

    except Exception as e:
        logger.error(f"Error loading pipeline and data: {str(e)}")
        raise


@app.on_event("startup")
async def startup_event():
    """Load model and data on startup"""
    load_pipeline_and_data()


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Store Sales Prediction API", "status": "healthy"}


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "pipeline_loaded": pipeline is not None,
        "reference_data_loaded": all(
            [
                holidays_df is not None,
                oil_df is not None,
                transactions_df is not None,
                stores_df is not None,
            ]
        ),
        "timestamp": datetime.now().isoformat(),
    }


def prepare_input_data(requests: List[PredictionRequest]) -> pd.DataFrame:
    """Convert prediction requests to DataFrame format expected by the pipeline"""
    data = []
    for req in requests:
        data.append(
            {
                "id": req.id,
                "date": pd.to_datetime(req.date),
                "store_nbr": req.store_nbr,
                "family": req.family,
                "onpromotion": req.onpromotion,
                "sales": np.nan,  # Initialize sales column as required by pipeline
            }
        )

    return pd.DataFrame(data)


def make_predictions(input_df: pd.DataFrame) -> np.ndarray:
    """Make predictions using the loaded pipeline"""
    try:
        if hasattr(pipeline, "named_steps"):
            # For sklearn-style pipelines
            # Transform the data through feature engineering
            transformed = pipeline.named_steps["feature_engineering"].transform(input_df)
            original_family = transformed["family"].copy()

            # Apply preprocessing
            processed = pipeline.named_steps["preprocessing"].transform(transformed)

            # Create final DataFrame for model prediction
            final_df = pd.DataFrame(processed, columns=transformed.columns.drop(["id", "date"]))
            final_df.rename(columns={"family": "family_encoded"}, inplace=True)
            final_df["family"] = original_family.values

            # Make predictions
            predictions = pipeline.named_steps["model_selector"].predict(final_df)
        else:
            # For simple model objects
            # Prepare minimal features for prediction
            features_df = input_df[["store_nbr", "family", "onpromotion"]].copy()
            predictions = pipeline.predict(features_df)

        return predictions

    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        raise


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(request: PredictionRequest):
    """Make a single prediction"""
    try:
        if pipeline is None:
            raise HTTPException(status_code=500, detail="Pipeline not loaded")

        # Convert single request to DataFrame
        input_df = prepare_input_data([request])

        # Make prediction
        predictions = make_predictions(input_df)

        return PredictionResponse(id=request.id, predicted_sales=float(predictions[0]))

    except Exception as e:
        logger.error(f"Error in single prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Make batch predictions"""
    try:
        if pipeline is None:
            raise HTTPException(status_code=500, detail="Pipeline not loaded")

        if not request.predictions:
            raise HTTPException(status_code=400, detail="No predictions requested")

        # Convert requests to DataFrame
        input_df = prepare_input_data(request.predictions)

        # Make predictions
        predictions = make_predictions(input_df)

        # Format response
        prediction_responses = [
            PredictionResponse(id=req.id, predicted_sales=float(pred))
            for req, pred in zip(request.predictions, predictions)
        ]

        return BatchPredictionResponse(
            predictions=prediction_responses, total_predictions=len(prediction_responses)
        )

    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.post("/predict/csv")
async def predict_from_csv(file: UploadFile = File(...)):
    """Make predictions from uploaded CSV file"""
    try:
        if pipeline is None:
            raise HTTPException(status_code=500, detail="Pipeline not loaded")

        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="File must be a CSV")

        # Read CSV file
        content = await file.read()
        input_df = pd.read_csv(pd.io.common.StringIO(content.decode("utf-8")))

        # Validate required columns
        required_columns = ["id", "date", "store_nbr", "family"]
        missing_columns = [col for col in required_columns if col not in input_df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=400, detail=f"Missing required columns: {missing_columns}"
            )

        # Add onpromotion column if missing
        if "onpromotion" not in input_df.columns:
            input_df["onpromotion"] = 0

        # Add sales column (required by pipeline)
        input_df["sales"] = np.nan

        # Convert date column
        input_df["date"] = pd.to_datetime(input_df["date"])

        # Make predictions
        predictions = make_predictions(input_df)

        # Create result DataFrame
        result_df = input_df[["id"]].copy()
        result_df["predicted_sales"] = predictions

        # Convert to JSON response
        results = result_df.to_dict("records")

        return JSONResponse(
            content={
                "predictions": results,
                "total_predictions": len(results),
                "status": "success",
            }
        )

    except Exception as e:
        logger.error(f"Error in CSV prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"CSV prediction failed: {str(e)}")


@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model"""
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline not loaded")

    try:
        info = {"model_type": str(type(pipeline)), "status": "loaded"}

        if hasattr(pipeline, "named_steps"):
            info["pipeline_steps"] = list(pipeline.named_steps.keys())
            if "feature_engineering" in pipeline.named_steps:
                fe_step = pipeline.named_steps["feature_engineering"]
                if hasattr(fe_step, "named_steps"):
                    info["feature_engineering_steps"] = list(fe_step.named_steps.keys())

        return info
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving model info: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
