from __future__ import annotations

import pandas as pd


class Plugin:
    plugin_params = {
        "input_data_file": "examples/data/eurusd.csv",
        "date_column": "DATE_TIME",
        "headers": True,
        "max_rows": None,
    }

    def __init__(self, config=None):
        self.params = self.plugin_params.copy()
        if config:
            self.set_params(**config)

    def set_params(self, **kwargs):
        self.params.update(kwargs)

    def load_data(self, config):
        file_path = config.get("input_data_file", self.params["input_data_file"])
        headers = bool(config.get("headers", self.params["headers"]))
        max_rows = config.get("max_rows", self.params["max_rows"])
        df = pd.read_csv(file_path, header=0 if headers else None, nrows=max_rows)

        date_col = config.get("date_column", self.params["date_column"])
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col]).set_index(date_col)
        return df
