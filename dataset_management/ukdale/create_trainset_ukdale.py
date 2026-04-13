"""Create training data for UK-DALE (2015) dataset."""

import argparse
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import pandas as pd

from ukdale_parameters import params_appliance

# Add parent directory to path to import functions
sys.path.append("dataset_management/")
from functions import load_dataframe  # pylint: disable=C0413 # noqa


DEFAULT_APPLIANCE = "dishwasher"

DATASET = "UK-DALE_2015"
DATA_DIR = "../../../data/"
INPUT_DATA_DIR = f"{DATA_DIR}/raw/{DATASET}/data/"
OUTPUT_DATA_DIR = f"{DATA_DIR}/processed/{DATASET}/{DEFAULT_APPLIANCE}/"

AGG_MEAN = 522
AGG_STD = 814


def get_arguments():
    """Get arguments from terminal."""
    parser = argparse.ArgumentParser(
        description="sequence to point learning example for NILM"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=INPUT_DATA_DIR,
        help="The directory containing the UKDALE data",
    )
    parser.add_argument(
        "--appliance_name",
        type=str,
        default=DEFAULT_APPLIANCE,
        help=f"Default: {DEFAULT_APPLIANCE}. Other: kettle, microwave, fridge, washingmachine",
    )
    parser.add_argument(
        "--aggregate_mean",
        type=int,
        default=AGG_MEAN,
        help="Mean value of aggregated reading (mains)",
    )
    parser.add_argument(
        "--aggregate_std",
        type=int,
        default=AGG_STD,
        help="Std value of aggregated reading (mains)",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=OUTPUT_DATA_DIR,
        help="The directory to store the training data",
    )
    return parser.parse_args()


args = get_arguments()
appliance_name = args.appliance_name
print(appliance_name)


def main():
    """Create training data for UK-DALE (2015) dataset."""
    start_time = time.time()
    sample_seconds = 8
    training_building_percent = 95  # Why?
    validation_percent = 13  # Why?
    debug = False

    train = pd.DataFrame(columns=["aggregate", appliance_name])
    Path(args.save_path).mkdir(parents=True, exist_ok=True)

    params = params_appliance[appliance_name]
    houses = params["houses"]
    channels = params["channels"]
    for h in houses:
        print(f"\t{args.data_dir}house_{h}/channel_{channels[houses.index(h)]}.dat")

        mains_df = load_dataframe(args.data_dir, h, 1)
        app_df = load_dataframe(
            args.data_dir,
            h,
            channels[houses.index(h)],
            col_names=["time", appliance_name],
        )

        mains_df["time"] = pd.to_datetime(mains_df["time"], unit="s")
        mains_df.set_index("time", inplace=True)
        mains_df.columns = ["aggregate"]
        # resample = mains_df.resample(str(sample_seconds) + 'S').mean()
        mains_df.reset_index(inplace=True)

        if debug:
            print(f"\tmains_df:\n{mains_df.head()}")
            plt.plot(mains_df["time"], mains_df["aggregate"])
            plt.show()

        # Appliance
        app_df["time"] = pd.to_datetime(app_df["time"], unit="s")

        if debug:
            print(f"\tapp_df:\n{app_df.head()}")
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

        del mains_df, app_df, df_align["time"]

        # plot the dataset
        if debug:
            print(f"\tdf_align:\n{df_align.head()}")
            plt.plot(df_align["aggregate"].values)
            plt.plot(df_align[appliance_name].values)
            plt.show()

        # Normalization
        mean = params["mean"]
        std = params["std"]

        df_align["aggregate"] = (
            df_align["aggregate"] - args.aggregate_mean
        ) / args.aggregate_std
        df_align[appliance_name] = (df_align[appliance_name] - mean) / std

        # Test CSV
        if h == params["test_build"]:
            df_align.to_csv(
                f"{args.save_path}{appliance_name}_test_.csv",
                mode="a",
                index=False,
                header=False,
            )
            print(f"\tSize of test set is {len(df_align) / 10**6:.4f} M rows.")
            continue

        # train = train.append(df_align, ignore_index=True)  # deprecated
        train = pd.concat([train, df_align], ignore_index=True)
        del df_align

    # Crop dataset
    if training_building_percent != 0:
        train.drop(
            train.index[-int((len(train) / 100) * training_building_percent) :],
            inplace=True,
        )

    # Validation CSV
    val_len = int((len(train) / 100) * validation_percent)
    val = train.tail(val_len)
    val.reset_index(drop=True, inplace=True)
    train.drop(train.index[-val_len:], inplace=True)
    val.to_csv(
        f"{args.save_path}{appliance_name}_validation_.csv",
        mode="a",
        index=False,
        header=False,
    )

    # Training CSV
    train.to_csv(
        f"{args.save_path}{appliance_name}_training_.csv",
        mode="a",
        index=False,
        header=False,
    )

    print(f"\tSize of total training set is {len(train) / 10**6:.4f} M rows.")
    print(f"\tSize of total validation set is {len(val) / 10**6:.4f} M rows.")
    print(f"\nPlease find files in: {args.save_path}")
    print(f"Total elapsed time: {(time.time() - start_time) / 60:.2f} min.")
    del train, val


if __name__ == "__main__":
    main()
