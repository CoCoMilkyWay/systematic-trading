from datetime import datetime
import time
from dateutil import relativedelta
import os

import backtrader as bt
import backtrader.feeds as btfeeds
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm


class MomentumStrategy(bt.Strategy):
    """
    A momentum strategy that goes long the top quantile of stocks
    and short the bottom quantile of stocks.
    """

    params = (
        ("long_quantile", 0.8),  # Long quantile threshold (e.g., top 80%)
        ("short_quantile", 0.2),  # Short quantile threshold (e.g., bottom 20%)
    )

    def __init__(self):
        self.ret = np.zeros(len(self.datas))

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f"{dt.isoformat()},{txt}")

    def is_first_business_day(self, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        first_day = dt.replace(day=1)
        first_business_day = pd.date_range(first_day, periods=1, freq="BMS")[0]
        return dt == first_business_day.date()

    def next(self):
        """
        Execute trades based on the momentum strategy.
        """
        if not self.is_first_business_day():
            return

        # Calculate returns for all stocks
        self.ret = np.array(
            [
                (d.close[-20] / d.close[-252] - 1 if len(d) > 252 else np.NaN)
                for d in self.datas
            ]
        )
        self.log(self.broker.getvalue())

        # Count the number of stocks that have a valid momentum predictor
        num_stocks = np.count_nonzero(~np.isnan(self.ret))

        # Compute the quantile thresholds
        long_threshold = np.nanquantile(self.ret, self.params.long_quantile)
        short_threshold = np.nanquantile(self.ret, self.params.short_quantile)

        for i, d in enumerate(self.datas):
            if self.ret[i] > long_threshold:  # Long the top quantile stocks
                self.order_target_percent(
                    data=d,
                    target=0.7 / num_stocks,
                )
            elif self.ret[i] < short_threshold:  # Short the bottom quantile stocks
                self.order_target_percent(
                    data=d,
                    target=-0.7 / num_stocks,
                )
            else:  # Close positions that don't meet the long or short criteria
                self.close(data=d)


class CashNav(bt.analyzers.Analyzer):
    """
    Analyzer returning cash and market values
    """

    def create_analysis(self):
        self.rets = {}
        self.vals = 0.0

    def notify_cashvalue(self, cash, value):
        self.vals = (
            self.strategy.datetime.datetime(),
            cash,
            value,
        )
        self.rets[len(self)] = self.vals

    def get_analysis(self):
        return self.rets


def main():
    path = os.path.join("/tmp/Chuyin980321", "momentum.pkl")
    if os.path.exists(path):
        print("Load Data: ", path)
        open(path).close()
        df = pickle.load(open(path, "rb"))
    else:
        print("Download Huggingface dataset to: ", path)
        dataset = load_dataset(
            "chuyin0321/timeseries-daily-stocks", split="train")
        print("Huggingface dataset Loaded: ", dataset)

        # df = pd.DataFrame(dataset)
        df = dataset.to_pandas()
        print("panda dataframe Loaded: ", df)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as file:
            pickle.dump(df, file)

    starting_date = datetime(1990, 1, 1)  # start from 1990

    print("\nDatabase Preparation")
    symbols = df["symbol"].unique()
    # symbols = symbols[np.random.choice(len(symbols), size=50, replace=False)]
    print(len(symbols), symbols)
    df["date"] = pd.to_datetime(df["date"])

    print("\nchecking Null value in Database")
    null_rows = df[df.isnull().any(axis=1) == True].copy()
    df = df[df.isnull().any(axis=1) == False].copy()
    print(null_rows)
    # null_stocks = null_rows["symbol"].unique()
    # print("stocks dropped: %s" % null_stocks)

    print("\nperform sector-wise analysis")
    # ['Health Care' 'Industrials' 'Information Technology' 'Financials' 'Consumer Staples' 'Utilities' 'Materials' 'Real Estate' 'Consumer Discretionary' 'Energy' 'Communication Services' '' 'Finance' 'Technology' 'Telecommunications' 'Basic Materials' 'Miscellaneous']    
    dataset = load_dataset("chuyin0321/perimeter-stocks", split="train")
    pr = dataset.to_pandas()
    sectors = pr["gics_sector"].unique()
    print(sectors)

    NUM_COLORS = len(sectors)
    colormap = plt.get_cmap('gist_rainbow')
    _, ax = plt.subplots()
    ax.set_prop_cycle('color', [colormap(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
    plt.ion()
    plt.show()

    # pr_sectors: perimeter table filtered by sector
    # df_symbol: datetimefile table filtered by symbol
    print("\nCreate and add data_feeds/strategy to Cerebro")
    for sector in (bar0 := tqdm(sectors)):
        bar0.set_description("%s: " % sector)
        cerebro = bt.Cerebro()
        cerebro.broker.set_cash(1000000)
        pr_sectors = pr[pr["gics_sector"] == sector].copy()
        sector_symbols = pr_sectors["symbol"]
        num_sym_in_sector = 0
        for _, symbol in enumerate(bar1 := tqdm(sector_symbols)):
            bar1.set_description("%s: " % symbol)

            # choose stocks
            if symbol not in symbols:
                continue
            if sector == "":
                continue
            df_symbol = df[df["symbol"] == symbol].copy()
            if df_symbol["date"].min() > starting_date:
                continue
            df_symbol = df_symbol[df_symbol["date"] > starting_date].copy()

            # adjust price
            factor = df_symbol["adj_close"] / df_symbol["close"]
            df_symbol["open"] = df_symbol["open"] * factor
            df_symbol["high"] = df_symbol["high"] * factor
            df_symbol["low"] = df_symbol["low"] * factor
            df_symbol["close"] = df_symbol["close"] * factor

            # feed data
            df_symbol.drop(["symbol", "adj_close"], axis=1, inplace=True)
            df_symbol.set_index("date", inplace=True)
            data = btfeeds.PandasData(dataname=df_symbol)
            cerebro.adddata(data, name=symbol)

            num_sym_in_sector += 1

        if num_sym_in_sector <= 1:  # this strategy needs at least 2 stocks
            continue
        cerebro.addstrategy(
            MomentumStrategy, long_quantile=0.8, short_quantile=0.2
        )  # Adjust parameters as desired
        cerebro.addanalyzer(CashNav, _name="cash_nav")

        bar0.set_description("Strategy Run")
        results = cerebro.run()
        print(sector, "final portfolio value: %.2f" %
              cerebro.broker.getvalue())

        dictionary = results[0].analyzers.getbyname(
            "cash_nav").get_analysis()
        df_plot = pd.DataFrame(dictionary).T
        df_plot.columns = ["Date", "Cash", "CashNav"]
        df_plot.set_index("Date", inplace=True)
        ax.plot(df_plot.loc[df_plot.index >= datetime(2010, 1, 1), ["CashNav"]], label=''.join(
            [sector, ": ", str(num_sym_in_sector)]))
        plt.gcf().canvas.draw()
        plt.savefig("./figure/momentum.pdf")  # save partial results
        plt.savefig("./figure/acce.png", dpi=1300)
        plt.legend()
        plt.pause(1)
    plt.savefig("./figure/momentum.pdf")
    plt.savefig("./figure/acce.png", dpi=1300)
    plt.show(block=True)


if __name__ == "__main__":
    main()
