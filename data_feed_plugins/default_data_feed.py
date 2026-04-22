"""
default_data_feed.py

Loads OHLCV data from CSV (pandas) and converts it to a backtrader
PandasData feed for the env.

Required CSV columns (case-insensitive): DATE_TIME + OPEN HIGH LOW CLOSE VOLUME.
Missing OHLCV columns are filled from CLOSE so backtrader is happy.
"""
from __future__ import annotations

from typing import Any, Dict

import backtrader as bt
import pandas as pd


class Plugin:
    plugin_params = {
        "input_data_file": "examples/data/eurusd_sample.csv",
        "date_column": "DATE_TIME",
        "headers": True,
        "max_rows": None,
        "price_column": "CLOSE",
    }

    def __init__(self, config: Dict[str, Any] | None = None):
        self.params = self.plugin_params.copy()
        if config:
            self.set_params(**config)

    def set_params(self, **kwargs: Any) -> None:
        self.params.update(kwargs)

    # ------------------------------------------------------------------
    def load_data(self, config: Dict[str, Any]) -> pd.DataFrame:
        file_path = config.get("input_data_file", self.params["input_data_file"])
        headers = bool(config.get("headers", self.params["headers"]))
        max_rows = config.get("max_rows", self.params["max_rows"])
        df = pd.read_csv(file_path, header=0 if headers else None, nrows=max_rows)

        date_col = config.get("date_column", self.params["date_column"])
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col]).set_index(date_col)

        # Fill missing OHLC with CLOSE so PandasData works
        price_col = config.get("price_column", self.params["price_column"])
        if price_col not in df.columns:
            raise ValueError(f"price_column '{price_col}' not found in data")
        for col in ("OPEN", "HIGH", "LOW", "CLOSE"):
            if col not in df.columns:
                df[col] = df[price_col]
        if "VOLUME" not in df.columns:
            df["VOLUME"] = 0
        return df

    def build_bt_feed(self, dataframe: pd.DataFrame, config: Dict[str, Any]) -> bt.feeds.PandasData:
        df = dataframe.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce")
            df = df.dropna()
        # backtrader expects lowercase default column names on PandasData; rename inline
        df = df.rename(
            columns={
                "OPEN": "open",
                "HIGH": "high",
                "LOW": "low",
                "CLOSE": "close",
                "VOLUME": "volume",
            }
        )
        for col in ("open", "high", "low", "close"):
            if col not in df.columns:
                raise ValueError(f"feed is missing column '{col}' after normalization")
        if "volume" not in df.columns:
            df["volume"] = 0
        df["openinterest"] = 0
        return bt.feeds.PandasData(dataname=df)
