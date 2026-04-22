import pandas as pd
from typing import Optional
import os as _os
_QUIET = _os.environ.get('PREDICTOR_QUIET', '0') == '1'


import pandas as pd
from typing import Optional
import sys

def load_csv(file_path: str, headers: bool = False, max_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Loads a CSV file with optional row limiting and processes it into a cleaned DataFrame.

    This function ensures consistent index handling by setting 'DATE_TIME' as the index
    if it exists in the dataset. If 'DATE_TIME' is missing, a RangeIndex is used, and
    warnings are logged.

    Args:
        file_path (str): Path to the CSV file.
        headers (bool): Whether the file contains headers. Defaults to False.
        max_rows (Optional[int]): Maximum number of rows to read. Defaults to None.

    Returns:
        pd.DataFrame: A processed DataFrame with numeric columns and a consistent index.

    Raises:
        Exception: Propagates any exception that occurs during the CSV loading process.
    """
    try:
        # 1) Load raw CSV data
        if headers:
            data = pd.read_csv(file_path, sep=',', dtype=str, nrows=max_rows)
        else:
            data = pd.read_csv(file_path, header=None, sep=',', dtype=str, nrows=max_rows)

        # 2) Detect 'DATE_TIME' column in a case-insensitive manner
        date_time_cols = [c for c in data.columns if c.strip().lower() == 'date_time']

        if date_time_cols:
            # 2a) Use the first detected 'DATE_TIME' column as the index
            main_dt_col = date_time_cols[0]
            data[main_dt_col] = pd.to_datetime(data[main_dt_col], errors='coerce')
            data.set_index(main_dt_col, inplace=True)

            # Drop extra 'DATE_TIME' columns if any
            extra_dt_cols = date_time_cols[1:]
            for c in extra_dt_cols:
                if c in data.columns:
                    data.drop(columns=[c], inplace=True, errors='ignore')

        else:
            # 2b) If 'DATE_TIME' is missing, use RangeIndex and log a warning
            if not _QUIET: print(f"Warning: No 'DATE_TIME' column found in '{file_path}'. Using RangeIndex.")
            data.index = pd.RangeIndex(start=0, stop=len(data), step=1)

        # 3) Rename columns if headers are missing
        if not headers:
            data.columns = [f'col_{i}' for i in range(len(data.columns))]

        # 4) Convert all columns to numeric, fill NaN values with 0
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

        # 5) Debug information
        if not _QUIET: print(f"[DEBUG] Loaded CSV '{file_path}' -> shape={data.shape}, index={data.index.dtype}, headers={headers}")

        # 6) Check for leftover NaNs
        if data.isnull().values.any():
            if not _QUIET: print(f"Warning: NaN values found after processing CSV: {file_path}")

    except Exception as e:
        if not _QUIET: print(f"An error occurred while loading the CSV: {e}")
        raise

    return data



def write_csv(file_path: str, data: pd.DataFrame, include_date: bool = True,
              headers: bool = True, window_size: Optional[int] = None) -> None:
    """
    Writes a DataFrame to a CSV file with optional date inclusion and headers.

    This function exports the provided DataFrame to a CSV file at the specified path.
    It allows for conditional inclusion of the date column and headers. An optional
    `window_size` parameter is present for future extensions but is not utilized in
    the current implementation.

    Args:
        file_path (str): The destination path for the CSV file.
        data (pd.DataFrame): The DataFrame to be written to the CSV.
        include_date (bool, optional): Determines whether to include the date column
            in the CSV. If `True` and the DataFrame contains a 'date' column, it is included
            as the index. Defaults to `True`.
        headers (bool, optional): Indicates whether to write the column headers to the CSV.
            Defaults to `True`.
        window_size (int, optional): Placeholder for windowing functionality.
            Not used in the current implementation. Defaults to `None`.

    Raises:
        Exception: Propagates any exception that occurs during the CSV writing process.

    Example:
        >>> write_csv("data/output.csv", df, include_date=True, headers=True)
    """
    try:
        if include_date and 'date' in data.columns:
            data.to_csv(file_path, index=True, header=headers)
        else:
            data.to_csv(file_path, index=False, header=headers)
    except Exception as e:
        if not _QUIET: print(f"An error occurred while writing the CSV: {e}")
        raise
