import sys
from typing import Tuple, Union
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import logging
import os
import shutil
from distutils.dir_util import copy_tree
from pathlib import Path
import albumentations as A
import cv2


def calculate_class_multiplier(
    class_df: pd.DataFrame,
    largest_class_size: int,
    global_multiplier=1.0,
) -> float:
    """
    Calculate multiplier with respect to the largest class.

    :param class_df: DataFrame with labels and paths for a single class.
    :param largest_class_size: Largest class size.
    :param global_multiplier: Global multiplier for every class.
    :return: Class multiplier.
    """
    return global_multiplier * largest_class_size / len(class_df)


def balance_class(
    df: pd.DataFrame,
    pipeline: A.Compose,
    data_path: str,
    subset: str,
    multiplier: float,
    label_col="label",
    filename_col="filename",
    seed: Union[int, None] = None,
) -> pd.DataFrame:
    """
    Balance class by augmenting it.

    :param df: DataFrame with labels and paths for a single class.
    :param data_path: Path to data directory.
    :param subset: Subset name.
    :param multiplier: Multiplier for class.
    :param label_col: Name of label column.
    :param filename_col: Name of filename column.
    :param seed: Random seed.
    :return: Augmented DataFrame.
    """
    augment_count = int(np.ceil(multiplier * len(df))) - len(df)
    df_sample = df.sample(n=augment_count, random_state=seed, replace=True)
    row: pd.Series[str]
    for _, row in df_sample.iterrows():
        filename = row[filename_col]
        label = row[label_col]
        image = cv2.imread(os.path.join(data_path, subset, label, filename))
        augmented = pipeline(image=image)
        augmented_image = augmented["image"]
        random_filename = f"aug_{np.random.randint(0, 1_000_000)}{filename}"
        df = pd.concat(
            [df, pd.DataFrame({label_col: [label], filename_col: [random_filename]})],
            ignore_index=True,
        )
        cv2.imwrite(
            os.path.join(data_path, subset, label, random_filename), augmented_image
        )
    return df


def copy_test(
    df: pd.DataFrame, data_path: str, label_col="label", filename_col="filename"
) -> None:
    """
    Copy test files to test subdirectory.

    :param df: DataFrame with labels and paths.
    :param data_path: Path to data directory.
    :param label_col: Name of label column.
    :param filename_col: Name of filename column.
    """
    row: pd.Series[str]
    for _, row in df.iterrows():
        filename = row[filename_col]
        label = row[label_col]
        train_image_path = os.path.join(data_path, "train", label, filename)
        test_path = os.path.join(data_path, "test", label)
        Path(test_path).mkdir(parents=True, exist_ok=True)
        shutil.copy2(train_image_path, test_path)


def delete_copied(
    df: pd.DataFrame, data_path: str, label_col="label", filename_col="filename"
) -> None:
    """
    Delete copied test files from train subdirectory.

    :param df: DataFrame with labels and paths.
    :param data_path: Path to data directory.
    :param label_col: Name of label column.
    :param filename_col: Name of filename column.
    """
    row: pd.Series[str]
    for _, row in df.iterrows():
        label = row[label_col]
        filename = row[filename_col]
        train_path = os.path.join(data_path, "train", label, filename)
        os.remove(train_path)


def split_data(
    df: pd.DataFrame,
    data_path: str,
    pipeline: A.Compose,
    train_split=0.8,
    balance: Tuple[bool, bool] = (False, False),
    label_col="label",
    filename_col="filename",
    global_multiplier=1.0,
    seed: Union[int, None] = None,
) -> None:
    """
    Split data into training and test sets and optionally balance classes.
    If balance is True and global_multiplier is greater than 1.0, then in
    addition to balancing classes, the training set will be globally augmented.
    If balance is True and global_multiplier is equal to 1.0, then only class
    balancing will be performed.

    :param df: DataFrame with labels and filenames.
    :param data_path: Path to data directory.
    :param train_split: Train split ratio.
    :param balance: Whether to balance classes in training and test set in a tuple (balance_train, balance_test).
    :param label_col: Name of label column.
    :param filename_col: Name of filename column.
    :param global_multiplier: Global multiplier for class size.
    :param seed: Random seed.
    """
    classes = df[label_col].unique()
    logging.info(f"Classes: {classes}")

    largest_class_size = 0
    largest_train_class_size = 0

    balance_train, balance_test = balance
    if balance_test or balance_train:
        counts = df[label_col].value_counts()
        largest_class_size = counts.max()
        largest_train_class_size = int(np.ceil(largest_class_size * train_split))
        logging.info(f"Global multiplier: {global_multiplier}")
        logging.info(f"Largest train class size: {largest_train_class_size}")

    train_df = pd.DataFrame(columns=[label_col, filename_col])
    test_df = pd.DataFrame(columns=[label_col, filename_col])

    c: str
    for c in classes:
        df_c = df[df[label_col] == c]
        train_class_df = df_c.sample(frac=train_split, random_state=seed)
        test_class_df = df_c.drop(train_class_df.index).astype(str)

        logging.info(f"Class {c}: {len(train_class_df)} train samples")
        logging.info(f"Class {c}: {len(test_class_df)} test samples")

        logging.info(
            f"Class {c}: copying {len(test_class_df)} test files from train to test subdirectory"
        )
        copy_test(
            test_class_df, data_path, label_col=label_col, filename_col=filename_col
        )
        logging.info(
            f"Class {c}: deleting {len(test_class_df)} copied test files from train subdirectory"
        )
        delete_copied(
            test_class_df, data_path, label_col=label_col, filename_col=filename_col
        )

        np.random.seed(seed)
        if balance_train:
            multiplier_train = calculate_class_multiplier(
                train_class_df,
                largest_train_class_size,
                global_multiplier=global_multiplier,
            )
            logging.info(f"Class {c}: multiplier_train = {multiplier_train}")

            if multiplier_train > 1.0:
                train_class_df = balance_class(
                    train_class_df,
                    pipeline,
                    data_path,
                    "train",
                    multiplier_train,
                    label_col=label_col,
                    filename_col=filename_col,
                    seed=seed,
                )
                logging.info(
                    f"Class {c}: augmented to {len(train_class_df)} train samples"
                )

        if balance_test:
            multiplier_test = calculate_class_multiplier(
                test_class_df,
                largest_class_size - largest_train_class_size,
                global_multiplier=global_multiplier,
            )
            logging.info(f"Class {c}: multiplier_test = {multiplier_test}")

            if multiplier_test > 1.0:
                test_class_df = balance_class(
                    test_class_df,
                    pipeline,
                    data_path,
                    "test",
                    multiplier_test,
                    label_col=label_col,
                    filename_col=filename_col,
                    seed=seed,
                )
                logging.info(
                    f"Class {c}: augmented to {len(test_class_df)} test samples"
                )
        train_df = pd.concat([train_df, train_class_df], ignore_index=True)
        test_df = pd.concat([test_df, test_class_df], ignore_index=True)

    logging.info(f"Total: {len(train_df)} train samples")
    logging.info(f"Total: {len(test_df)} test samples")

    train_csv_path = os.path.join(data_path, "train.csv")
    test_csv_path = os.path.join(data_path, "test.csv")

    logging.info(
        f"Saving train DataFrame to {train_csv_path} with columns '{label_col}' and '{filename_col}'"
    )
    train_df.to_csv(train_csv_path, index=False)

    logging.info(
        f"Saving test DataFrame to {test_csv_path} with columns '{label_col}' and '{filename_col}'"
    )
    test_df.to_csv(test_csv_path, index=False)


def load_dataframe(
    train_path: str, label_col="label", filename_col="filename"
) -> pd.DataFrame:
    """
    Load DataFrame with labels and filenames.

    :param train_path: Path to train subdirectory.
    :param label_col: Name of label column.
    :param filename_col: Name of filename column.
    :return: DataFrame with labels and filenames.
    """
    logging.info(
        f"Creating DataFrame of labels and filenames from {train_path} with columns '{label_col}' and '{filename_col}'"
    )
    df = pd.DataFrame(columns=[label_col, filename_col])
    for label in os.listdir(train_path):
        label_path = os.path.join(train_path, label)
        for filename in os.listdir(label_path):
            df = pd.concat(
                [df, pd.DataFrame({label_col: [label], filename_col: [filename]})],
                ignore_index=True,
            )
    return df


def copy_to_output(from_path: str, to_path: str) -> None:
    """
    Copy files from one directory to another.

    :param from_path: Path to a directory with files.
    :param to_path: Path to a directory where files will be copied.
    """
    logging.info(f"Copying files from {from_path} to {to_path}")
    copy_tree(from_path, to_path)


def get_pipeline(yaml_path: Union[str, None] = None) -> A.Compose:
    """
    Get pipeline for augmentation.

    :param yaml_path: Path to a custom pipeline serialized to YAML. Can be None to use the default pipeline.
    :return: Albumentations Compose pipeline.
    """
    if yaml_path is not None:
        if not os.path.exists(yaml_path):
            logging.error(f"Custom pipeline YAML file {yaml_path} does not exist")
            sys.exit(1)
        logging.info(f"Loading custom pipeline from {yaml_path}")
        loaded = A.load(yaml_path, data_format="yaml")
        if not isinstance(loaded, A.Compose):
            logging.error(
                f"Pipeline loaded from {yaml_path} is not an instance of {A.Compose.__module__}.{A.Compose.__name__}"
            )
            sys.exit(1)
        return loaded

    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                always_apply=True, contrast_limit=0.2, brightness_limit=0.2
            ),
            A.OneOf(
                [
                    A.MotionBlur(always_apply=True),
                    A.GaussNoise(always_apply=True),
                    A.GaussianBlur(always_apply=True),
                ],
                p=0.5,
            ),
            A.PixelDropout(p=0.25),
            A.Rotate(always_apply=True, limit=20, border_mode=cv2.BORDER_REPLICATE),
        ]
    )


def create_cli() -> ArgumentParser:
    """
    Create CLI.

    :return: ArgumentParser.
    """
    parser = ArgumentParser(
        description="Split data into training and test sets and optionally balance and augment classes."
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to a directory that includes a train directory with the images in subdirectories named after the labels",
    )
    parser.add_argument(
        "--balance-train",
        action="store_true",
        help="Balance classes in training set and optionally perform global augmentation for the training set if GLOBAL_MULTIPLIER is set to greater than 1.0 (default: False)",
    )
    parser.add_argument(
        "--balance-test",
        action="store_true",
        help="Balance classes in created test set and optionally perform global augmentation for the test set if GLOBAL_MULTIPLIER is set to greater than 1.0 (default: False)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to an empty output directory (default: None - overwrite input directory)",
    )
    parser.add_argument(
        "--train-split",
        type=lambda x: float(x)
        if float(x) > 0.5
        else parser.error(
            "Train split ratio must be a floating point number greater than 0.5"
        ),
        default=0.8,
        help="Train split ratio (default: 0.8)",
    )
    parser.add_argument(
        "--seed",
        type=lambda x: int(x)
        if int(x) > 0
        else parser.error("Seed must be an integer greater than 0"),
        default=None,
        help="Random seed (default: None)",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="label",
        help="Name of label column you want to be created in the CSV files (default: 'label')",
    )
    parser.add_argument(
        "--filename-col",
        type=str,
        default="filename",
        help="Name of filename column you want to be created in the CSV files (default: 'filename')",
    )
    parser.add_argument(
        "--global-multiplier",
        type=lambda x: float(x)
        if float(x) >= 1.0
        else parser.error(
            "Multiplier must be a floating point number greater than or equal to 1.0"
        ),
        default=1.0,
        help="Global multiplier for the number of images in each class (default: 1.0). This option can be used to increase the number of images in each class but is ignored if --balance is not used.",
    )
    parser.add_argument(
        "--pipeline-yaml",
        type=str,
        default=None,
        help="Path to a custom Albumentations Compose pipeline serialized to YAML (default: None - use pipeline included in this script)",
    )
    return parser


def main() -> None:
    cli = create_cli()
    args = cli.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    path = args.path
    if not os.path.exists(path):
        logging.error(f"Path {path} does not exist.")
        sys.exit(1)

    train_path = os.path.join(path, "train")
    if not os.path.exists(train_path):
        logging.error(f"Path {path} does not include train subdirectory.")
        sys.exit(1)

    pipeline = get_pipeline(args.pipeline_yaml)

    if args.output_path is not None:
        old_train_path = train_path
        path = args.output_path
        train_path = os.path.join(args.output_path, "train")
        Path(train_path).mkdir(parents=True, exist_ok=True)
        copy_to_output(old_train_path, train_path)

    df = load_dataframe(
        train_path, label_col=args.label_col, filename_col=args.filename_col
    )

    split_data(
        df,
        path,
        pipeline,
        train_split=args.train_split,
        balance=(args.balance_train, args.balance_test),
        label_col=args.label_col,
        filename_col=args.filename_col,
        global_multiplier=args.global_multiplier,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
