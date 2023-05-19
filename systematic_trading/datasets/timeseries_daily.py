"""
Timeseries daily data.
"""
from dataset import Dataset
import pandas as pd


class TimeseriesDaily(Dataset):
    """
    Timeseries daily data.
    """

    def __init__(self):
        super().__init__()
        self.expected_columns = [
            "symbol",
            "date",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
        ]
        self.data = pd.DataFrame(columns=self.expected_columns)
