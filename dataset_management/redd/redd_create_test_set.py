import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from create_trainset_redd import DEFAULT_APPLIANCE, INPUT_DATA_DIR, OUTPUT_DATA_DIR


def window_stack(a, stepsize=1, width=3):
    """Returns a 2D array where the i-th row is a window of the input array."""
    return np.hstack(a[i : 1 + i - width or None : stepsize] for i in range(0, width))


def remove_space(string):
    """Removes spaces from a string. Used for parsing terminal inputs."""
    return string.replace(" ", "")


params_appliance = {
    "microwave": {
        "windowlength": 599,
        "on_power_threshold": 200,
        "max_on_power": 3969,
        "mean": 500,
        "std": 800,
        "s2s_length": 128,
        "houses": [1, 2, 3],
        "channels": [11, 6, 16],
    },
    "fridge": {
        "windowlength": 599,
        "on_power_threshold": 50,
        "max_on_power": 3323,
        "mean": 200,
        "std": 400,
        "s2s_length": 512,
        "houses": [1, 2, 3],
        "channels": [5, 9, 7, 18],
    },
    "dishwasher": {
        "windowlength": 599,
        "on_power_threshold": 10,
        "max_on_power": 3964,
        "mean": 700,
        "std": 1000,
        "s2s_length": 1536,
        "houses": [1, 2, 3],
        "channels": [6, 10, 9],
    },
    "washingmachine": {
        "windowlength": 599,
        "on_power_threshold": 20,
        "max_on_power": 3999,
        "mean": 400,
        "std": 700,
        "s2s_length": 2000,
        "houses": [1, 2, 3],
        "channels": [20, 7, 13],
    },
}


start_time = time.time()
app_list = ["microwave", "fridge", "dishwasher", "washingmachine"]


sample_seconds = 8  # fixed
debug = False
nrows = None

for app in app_list:
    appliance_name = app
    print(f"Appliance name: {appliance_name}")

    params = params_appliance[appliance_name]
    houses = params["houses"]
    channels = params["channels"]
    for h in houses:
        print(f"{INPUT_DATA_DIR}house_{h}/channel_{channels[houses.index(h)]}.dat")

        # read data
        mains1_df = pd.read_table(
            f"{INPUT_DATA_DIR}house_{h}/channel_{1}.dat",
            sep="\\s+",
            nrows=nrows,
            usecols=[0, 1],
            names=["time", "mains1"],
            dtype={"time": str},
        )

        mains2_df = pd.read_table(
            f"{INPUT_DATA_DIR}house_{h}/channel_{2}.dat",
            sep="\\s+",
            nrows=nrows,
            usecols=[0, 1],
            names=["time", "mains2"],
            dtype={"time": str},
        )
        app_df = pd.read_table(
            f"{INPUT_DATA_DIR}house_{h}/channel_{channels[houses.index(h)]}.dat",
            sep="\\s+",
            nrows=nrows,
            usecols=[0, 1],
            names=["time", app],
            dtype={"time": str},
        )

        # Aggregate
        # mains1_df = mains1_df.set_index(mains1_df.columns[0])
        # mains1_df.index = pd.to_datetime(mains1_df.index, unit='s')
        mains1_df["time"] = pd.to_datetime(mains1_df["time"], unit="s")

        # mains2_df = mains2_df.set_index(mains2_df.columns[0])
        # mains2_df.index = pd.to_datetime(mains2_df.index, unit='s')
        mains2_df["time"] = pd.to_datetime(mains2_df["time"], unit="s")

        # merging two mains
        mains1_df.set_index("time", inplace=True)
        mains2_df.set_index("time", inplace=True)
        # mains_df = mains1_df.join(mains2_df, how='outer').interpolate(method='time')
        mains_df = mains1_df.join(mains2_df, how="outer")
        mains_df["aggregate"] = mains_df.iloc[:].sum(axis=1)

        resample = mains_df.resample(str(sample_seconds) + "S").mean()
        # mains_df = resample.mean()
        mains_df.reset_index(inplace=True)

        # resampling 8 sec
        # resample = mains_df.resample(str(sample_seconds)+'S')
        # mains_df = resample.mean()
        # ind = pd.date_range(0, periods=df.shape[0], freq='6S')
        # df.set_index(ind, inplace=True, drop=True)

        # deleting original separate mains
        del mains_df["mains1"], mains_df["mains2"]

        if debug:
            print(f"mains_df: {mains_df.head()}")
            plt.plot(mains_df["time"], mains_df["aggregate"])
            plt.show()

        # Appliance
        # app_df = app_df.set_index(app_df.columns[0])
        # app_df.index = pd.to_datetime(app_df.index, unit='s')
        app_df["time"] = pd.to_datetime(app_df["time"], unit="s")
        # app_df.columns = [appliance_name]
        if debug:
            print(f"app_df: {app_df.head()}")
            plt.plot(app_df["time"], app_df[appliance_name])
            plt.show()

        # the timestamps of mains and appliance are not the same, we need to align them
        # 1. join the aggregate and appliance dataframes;
        # 2. interpolate the missing values;
        mains_df.set_index("time", inplace=True)
        app_df.set_index("time", inplace=True)

        df_align = (
            mains_df.join(app_df, how="outer")
            .resample(str(sample_seconds) + "S")
            .mean()
            .fillna(method="backfill", limit=1)
        )
        df_align = df_align.dropna()

        df_align.reset_index(inplace=True)
        print(df_align.count())
        del mains1_df, mains2_df, mains_df, app_df, df_align["time"]

        mains = df_align["aggregate"].values
        app_data = df_align[appliance_name].values
        # plt.plot(np.arange(0, len(mains)), mains, app_data)
        # plt.show()

        # plot the dataset
        if debug:
            print(f"df_align: {df_align.head()}")
            plt.plot(df_align["aggregate"].values)
            plt.plot(df_align[appliance_name].values)
            plt.show()

        # Normalization
        aggregate_mean = 522
        aggregate_std = 814
        mean = params["mean"]
        std = params["std"]

        df_align["aggregate"] = (df_align["aggregate"] - aggregate_mean) / aggregate_std
        df_align[appliance_name] = (df_align[appliance_name] - mean) / std

        # Save to csv
        df_align.to_csv(f"{OUTPUT_DATA_DIR}{appliance_name}_test_redd_H{h}.csv", index=False)
        print(f"Size of test set is {df_align.shape[0] / 10**6:.3f} M rows (House {h}).")

        del df_align


print("\nNormalization parameters: ")
print("Mean and standard deviation values USED for AGGREGATE are:")
print(f"\tMean = {aggregate_mean:d}, STD = {aggregate_std:d}")

print("Mean and standard deviation values USED for " + appliance_name + " are:")
print(f"\tMean = {params['mean']:d}, STD = {params['std']:d}")
print("\nPlease find files in: " + OUTPUT_DATA_DIR)
print(f"\nTotal elapsed time: {int(int(time.time() - start_time) / 60)} min")
