"""
inference_pipeline.py

This script performs inference on new data using the best trained model from MLflow Model Registry.

Usage:
    python inference_pipeline.py --predict-month YYYY-MM

Arguments:
    --predict-month (str): The month for the dataset to predict on, in the format YYYY-MM.
"""

import argparse
import logging
import os
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.pipeline import Pipeline  # type: ignore

from datahandler import DataHandler
from mlflowhelper import MlFlowContext, MlFlowModelManager

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define paths
FILE_PATH = Path(__file__).resolve()
BASE_PATH = FILE_PATH.parent

# Load config
import yaml

config_path = os.path.join(BASE_PATH, "config.yaml")
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

DATA_DIR = (BASE_PATH.parent / "data").resolve()
S3_URL = config["constants"]["S3_URL"]
PREFIX = config["constants"]["PREFIX"]
HPO_CHAMPION_MODEL = config["constants"]["HPO_CHAMPION_MODEL"]
HPO_EXPERIMENT_NAME = config["constants"]["HPO_EXPERIMENT_NAME"]

# MLflow setup
TRACKING_URI = f"sqlite:///{BASE_PATH.parent / 'mlflow/mlflow.db'}"
mlflow.set_tracking_uri(TRACKING_URI)


def run_inference(predict_month: str):
    logger.info(f"Running inference for month: {predict_month}")

    # Download data
    file_name = PREFIX + predict_month + ".parquet"
    data_handler = DataHandler(DATA_DIR, S3_URL)
    file_path = data_handler.download_data(file_name)

    # Load and preprocess data
    df_raw = pd.read_parquet(file_path)
    df = data_handler.preprocess_dataset(df_raw)
    df = df.drop(columns=["is_cash_payment"])

    # Drop label if present
    if "target" in df.columns:
        df = df.drop(columns=["target"])

    # Dummy args for MLflow context
    from argparse import Namespace
    args_init = Namespace(
        train_month=None,
        val_month=None,
        test_month=None,
        num_trials=1,
        flag_reset_mlflow="N",  # Required to avoid error
    )

    # Load model from MLflow Model Registry
    mlflowcontext = MlFlowContext(TRACKING_URI.replace("sqlite://", ""), HPO_EXPERIMENT_NAME, args_init)
    model_manager = MlFlowModelManager(mlflowcontext)
    model_version: Pipeline = model_manager.get_production_version(HPO_CHAMPION_MODEL)
    model = model_manager.load_model_from_version(model_version)

    # Perform predictions
    predictions = model.predict(df)

    # Output predictions
    df_result = df.copy()
    df_result["prediction"] = predictions

    output_path = BASE_PATH / f"predictions_{predict_month}.csv"
    df_result.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference using registered model")
    parser.add_argument("--predict-month", type=str, required=True, help="Month to predict in format YYYY-MM")
    args = parser.parse_args()

    run_inference(args.predict_month)