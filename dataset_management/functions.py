"""Functions for dataset management."""

from pathlib import Path

import pandas as pd


def load_dataframe(
    directory, building, channel, col_names=["time", "data"], nrows=None
):
    df = pd.read_table(
        Path(directory, f"house_{building}", f"channel_{channel}.dat"),
        sep="\\s+",
        nrows=nrows,
        usecols=[0, 1],
        names=col_names,
        dtype={"time": str},
    )
    return df
