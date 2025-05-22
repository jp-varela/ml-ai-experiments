from datetime import datetime, timedelta
import logging
import os
from typing import Dict, List, Union
import yaml

import mlflow
import pandas as pd
from sklearn.pipeline import Pipeline

from datadrift import DataDrift
from datahandler import DataHandler
from mlflowhelper import MlFlowContext, MlFlowModelManager

FILE_PATH = os.path.abspath(__file__)
BASE_PATH = FILE_PATH.parent

config_path = os.path.join(BASE_PATH, "config.yaml")
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

DATA_DIR = (BASE_PATH.parent / "data").resolve()
S3_URL = config["constants"]["S3_URL"]
PREFIX = config["constants"]["PREFIX"]
HPO_CHAMPION_MODEL = config["constants"]["HPO_CHAMPION_MODEL"]
HPO_EXPERIMENT_NAME = config["constants"]["HPO_EXPERIMENT_NAME"]

TRACKING_URI = f"sqlite:///{BASE_PATH.parent / 'mlflow/mlflow.db'}"
mlflow.set_tracking_uri(TRACKING_URI)


class MyAssetWithQualityCheck(object):
    def __init__(self, input_tables: List[str]):
        logging.basicConfig(level=logging.INFO)

        self._input_tables = input_tables
        self._logger = logging.getLogger(__name__)

    def load_input_table(self, table_name: str, available_time: str) -> pd.DataFrame:
        file_name = table_name + available_time + ".parquet"
        data_handler = DataHandler(DATA_DIR, S3_URL)
        file_path = data_handler.download_data(file_name)

        df_raw = pd.read_parquet(file_path)
        df = data_handler.preprocess_dataset(df_raw)
        df = df.drop(columns=["is_cash_payment"])
        return df
        
    def _check_table_quality(self, curr_table, ref_table) -> bool:
        data_drift = DataDrift()
        quality_results = data_drift.check_quality(curr_table, ref_table)

        for column, result in quality_results.items():
            if result != "No drift detected":
                self._logger.warning(f"Data drift detected in column {column}: {result}")
                return False
        return True
        
    def _load_input_tables_with_quality(
            self, available_time: str, ref_table, min_available_times_map: Dict[str, str]
        ) -> Dict[str, pd.DataFrame]:
        """
        Load input tables and check their quality. For any table, if the quality check fails, it will try to load the
        previous day's data until the minimum available time is reached. If the quality check fails for all available
        data, the table will be set to None in the output dictionary.

        :param available_time: The date for which the data is requested in YYYYMMDD format.
        :param min_available_times_map: A dictionary mapping table names to their minimum available times in YYYYMMDD
                                        format.

        :return: A dictionary mapping table names to their loaded data. If a table's quality check fails, it will be set
                 to None.
        """
        input_tables_dict = {}
        for table_name in self._input_tables:
            table_ok = False
            available_time_date = datetime.strptime(available_time, "%Y%m%d")

            while not table_ok:
                curr_table = self.load_input_table(table_name, available_time_date.strftime("%Y-%m"))
                
                table_ok = self._check_table_quality(curr_table, ref_table)

                if not table_ok:
                    available_time_date -= timedelta(days=1)
                    if available_time_date < min_available_times_map[table_name].strptime("%Y%m%d"):
                        break
            if not table_ok:
                self._logger.warn(f"Table {table_name} is not available for the given time period.")
                input_tables_dict[table_name] = None
            else:
                input_tables_dict[table_name] = curr_table
        return input_tables_dict


    def transform(self, available_time: str, min_available_times_map: Dict[str, str]) -> pd.DataFrame:
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
        
        artifact_path = "data/temp.csv"
        
        # Download the artifact
        artifact_local_path = mlflow.artifacts.download_artifacts(
            run_id=model_version.run_id,
            artifact_path=artifact_path
        )
        
        df = pd.read_csv(artifact_local_path)

        tables_dict = self._load_input_tables_with_quality(
            available_time=available_time, ref_table=df,
            min_available_times_map=min_available_times_map
        )
        return tables_dict
    
    
if __name__ == "__main__":
    asset = MyAssetWithQualityCheck(input_tables=["green_tripdata_"])
    tables = asset.transform(available_time="202103", min_available_times_map={"green_tripdata_": "202101"})
    asset._logger.info(f"Loaded tables: {tables}")
    
    