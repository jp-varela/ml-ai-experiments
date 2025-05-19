from evidently.test_suite import TestSuite
from evidently.tests import TestColumnDrift
import pandas as pd

class DataDrift:
    def __init__(self, threshold: float = 0.05):
        """
        Initializes the DataDrift class.

        Args:
            threshold (float): The p-value threshold for drift detection. Default is 0.05.
        """
        self.threshold = threshold

    def check_quality(self, df1: pd.DataFrame, df2: pd.DataFrame) -> dict:
        """
        Checks for data drift between two dataframes using Evidently.

        Args:
            df1 (pd.DataFrame): Reference dataframe.
            df2 (pd.DataFrame): Current dataframe.

        Returns:
            dict: A dictionary with drift results for each column.
        """
        drift_results = {}
        for column in df1.columns:
            if column in df2.columns:
                test_suite = TestSuite(tests=[TestColumnDrift(column_name=column, threshold=self.threshold)])
                test_suite.run(reference_data=df1, current_data=df2)
                drift_results[column] = test_suite.as_dict()["tests"][0]["status"]
            else:
                drift_results[column] = "Column missing in current data"

        return drift_results