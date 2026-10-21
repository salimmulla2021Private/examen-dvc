
from pathlib import Path
from typing import IO

import pandas as pd


def load_data(data_path: str | Path | IO[bytes]) -> pd.DataFrame:
    """Load data from a CSV file."""
    return pd.read_csv(data_path)
