"""Create test data for UK-DALE dataset."""

import time
from pathlib import Path

import pandas as pd

from create_trainset_ukdale import DEFAULT_APPLIANCE, INPUT_DATA_DIR, OUTPUT_DATA_DIR


def load(data_dir, building, appliance, channel, n_rows=None):
    """Loads the csv file for a given building and channel."""
    file_name = Path(data_dir) / f"house_{building}" / f"channel_{channel}.dat"
    print(f"Loading data from {file_name}...")
    single_csv = pd.read_csv(
        file_name,
        sep=" ",
        names=["time", appliance],
        dtype={"time": str, "appliance": int},
        nrows=n_rows,
        usecols=[0, 1],
        engine="python",
    )
    return single_csv


start_time = time.time()
print(f"appliance_name: {DEFAULT_APPLIANCE}")

# UK-DALE path
path = Path(INPUT_DATA_DIR)
save_path = Path(OUTPUT_DATA_DIR)

aggregate_mean = 522
aggregate_std = 814
nrows = 10**5

params_appliance = {
    "kettle": {
        "windowlength": 599,
        "on_power_threshold": 2000,
        "max_on_power": 3998,
        "mean": 700,
        "std": 1000,
        "s2s_length": 128,
        "houses": [1, 2],
        "channels": [10, 8],
    },
    "microwave": {
        "windowlength": 599,
        "on_power_threshold": 200,
        "max_on_power": 3969,
        "mean": 500,
        "std": 800,
        "s2s_length": 128,
        "houses": [1, 2],
        "channels": [13, 15],
    },
    "fridge": {
        "windowlength": 599,
        "on_power_threshold": 50,
        "max_on_power": 3323,
        "mean": 200,
        "std": 400,
        "s2s_length": 512,
        "houses": [1, 2],
        "channels": [12, 14],
    },
    "dishwasher": {
        "windowlength": 599,
        "on_power_threshold": 10,
        "max_on_power": 3964,
        "mean": 700,
        "std": 1000,
        "s2s_length": 1536,
        "houses": [1, 2],
        "channels": [6, 13],
    },
    "washingmachine": {
        "windowlength": 599,
        "on_power_threshold": 20,
        "max_on_power": 3999,
        "mean": 400,
        "std": 700,
        "s2s_length": 2000,
        "houses": [1, 2],
        "channels": [5, 12],
    },
}

print("Starting creating testset...")

params = params_appliance[DEFAULT_APPLIANCE]
houses = params["houses"]
channels = params["channels"]

for house_id in houses:
    agg_df = load(path, house_id, DEFAULT_APPLIANCE, 1, n_rows=nrows)
    df = load(
        path,
        house_id,
        DEFAULT_APPLIANCE,
        channels[houses.index(house_id)],
        n_rows=nrows,
    )
    print(df.head(), agg_df.head())

    # Time conversion
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    agg_df["time"] = pd.to_datetime(agg_df["time"], unit="ms")

    df["aggregate"] = agg_df[DEFAULT_APPLIANCE]
    cols = df.columns.tolist()
    del cols[0]
    cols = cols[-1:] + cols[:-1]
    df = df[cols]

    # Re-sampling
    ind = pd.date_range(0, periods=df.shape[0], freq="6S")
    df.set_index(ind, inplace=True, drop=True)
    resample = df.resample("8S")
    df = resample.mean()

    # Normalization
    df["aggregate"] = (df["aggregate"] - aggregate_mean) / aggregate_std
    df[DEFAULT_APPLIANCE] = (df[DEFAULT_APPLIANCE] - params["mean"]) / params["std"]

    # Save
    save_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        f"{save_path}{DEFAULT_APPLIANCE}_test_uk-dale_H{house_id}.csv", index=False
    )
    print(f"Size of test set is {df.shape[0] / 10**6:.3f} M rows (House {house_id}).")


print("\nNormalization parameters: ")
print("Mean and standard deviation values USED for AGGREGATE are:")
print(f"\tMean = {aggregate_mean:d}, STD = {aggregate_std:d}")

print(f"Mean and standard deviation values USED for {DEFAULT_APPLIANCE} are:")
print(f'\tMean = {params["mean"]:d}, STD = {params["std"]:d}')

print(f"\nPlease find files in: {save_path}")
print(f"\nTotal elapsed time: {int(int(time.time() - start_time) / 60)} min")
