"""
Short interest from Nasdaq.
"""

from datetime import date, datetime
import time
import urllib

from datasets import load_dataset
import numpy as np
import pandas as pd
import requests

from systematic_trading.datasets.raw import Raw
from systematic_trading.helpers import nasdaq_headers, retry_get


class ShortInterest(Raw):
    """
    Short interest from Nasdaq.
    """

    def __init__(self, suffix: str = None, tag_date: date = None, username: str = None):
        super().__init__(suffix, tag_date, username)
        self.name = f"short-interest-{suffix}"
        self.expected_columns = [
            "symbol",
            "date",
            "id",
            "settlement_date",
            "interest",
            "avg_daily_share_volume",
            "days_to_cover",
        ]
        self.dataset_df = pd.DataFrame(columns=self.expected_columns)

    def append_frame(self, symbol: str):
        ticker = self.symbol_to_ticker(symbol)
        url = f"https://api.nasdaq.com/api/quote/{ticker}/short-interest?assetClass=stocks"
        try:
            response = retry_get(url, headers=nasdaq_headers(), mode="curl")
        except:
            self.frames[symbol] = None
            return
        json_data = response.json()
        if json_data["data"] is None:
            self.frames[symbol] = None
            return
        short_interest_table = json_data["data"]["shortInterestTable"]
        if short_interest_table is None:
            self.frames[symbol] = None
            return
        data = short_interest_table["rows"]
        df = pd.DataFrame(data=data)
        df["settlementDate"] = pd.to_datetime(df["settlementDate"])
        df["interest"] = df["interest"].apply(
            lambda x: int(x.replace(",", "")) if x != "N/A" else None
        )
        df["avgDailyShareVolume"] = df["avgDailyShareVolume"].apply(
            lambda x: int(x.replace(",", "")) if x != "N/A" else None
        )
        df.rename(
            columns={
                "settlementDate": "settlement_date",
                "avgDailyShareVolume": "avg_daily_share_volume",
                "daysToCover": "days_to_cover",
            },
            inplace=True,
        )
        df["id"] = range(len(df))
        df["symbol"] = symbol
        df["date"] = self.tag_date.isoformat()
        df = df.reindex(columns=self.expected_columns)
        self.frames[symbol] = df

    def set_dataset_df(self):
        self.dataset_df = pd.concat([f for f in self.frames.values() if f is not None])
        if self.check_file_exists():
            self.add_previous_data()
        self.dataset_df.sort_values(by=["symbol", "date", "id"], inplace=True)
        self.dataset_df.reset_index(drop=True, inplace=True)


if __name__ == "__main__":
    symbol = "AAPL"
    suffix = "stocks"
    tag_date = datetime(2023, 5, 26).date()
    username = "chuyin0321"
    dataset = ShortInterest(suffix=suffix, tag_date=tag_date, username=username)
    dataset.append_frame(symbol)
