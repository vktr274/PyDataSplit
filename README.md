# PyDataSplit - Easy Image Dataset Splitting and Balancing

The script requires installation of Python packages listed in the [`requirements.txt`](./requirements.txt) file. The packages can be installed using the following command:

`pip install -r requirements.txt`

Tested with Python 3.7.9.

## The Script

PyDataSplit is a Python script that splits your image dataset into training and test sets by random sampling from the original dataset and optionally balances the dataset by augmenting classes smaller in size relative to the largest class. If balancing is enabled, it can also optionally perform global augmentation. Meaning that the number of images in each class can be increased by a global multiplier. The script also creates a CSV file with label to filename mappings for the training and test sets.

The augmentation pipeline used it the script is created using the [Albumentations](https://albumentations.ai/) library. The pipeline is a composition of transformations that are applied to the images and is defined like this:

```py
A.Compose(
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
```

Horizontal flip is applied with a probability of 50%. Random brightness and contrast are always applied with a contrast limit of &pm;20% and a brightness limit of &pm;20%. One of motion blur, Gaussian noise, or Gaussian blur is applied with a probability of 50%. Pixel dropout is applied with a probability of 25%. Rotation by a random angle is always applied with a limit of &pm;20 degrees and border mode set to replicate colors at the borders of the image being rotated to avoid black borders.

If enabled, augmentations are applied both to the training and test sets.

Before running the script, make sure that the dataset is downloaded and extracted to a folder with the following structure:

```text
your_dataset_folder
└── train
    ├── class1
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── class2
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── ...
```

The script will create a `test` subdirectory in the directory and will move part of the images from the `train` subdirectory to the `test` subdirectory. This means that the default behavior of the script is to overwrite the input directory. The script will also create a `train.csv` and `test.csv` files in the directory. The `train.csv` file will contain the label to filename mappings for the training set and the `test.csv` file will contain the label to filename mappings for the test set.

If the script is ran with a specified output path, the script will first copy the images from the `train` subdirectory in the input directory to the `train` subdirectory in the output directory and will create the `test` subdirectory in the output directory. The `train.csv` and `test.csv` files will be created in the output directory. If the output directory does not exist, it will be created. If the output directory exists, it can only contain an empty `train` subdirectory or can be empty completely.

The script is universal and can be used for any dataset that has the same structure (dataset with a train subdirectory with images in subdirectories named after the labels).

## Running the Script

The script is available in the [`src/datasplit.py`](./src/datasplit.py) file. It can be run following this pattern:

`python datasplit.py [-h] [--balance-train] [--balance-test] [--output-path OUTPUT_PATH] [--train-split TRAIN_SPLIT] [--seed SEED] [--label-col LABEL_COL] [--filename-col FILENAME_COL] [--global-multiplier GLOBAL_MULTIPLIER] [--pipeline-yaml PIPELINE_YAML] path`

Positional argument:

- `path` - Path to a directory that includes a train directory with the images in subdirectories named after the labels, e.g. if `path` is `data`, then the images should be in `data/train/class1`, `data/train/class2`, etc.

Options:

- `-h`, `--help` - show help message and exit
- `--balance-train` - Balance classes in training set and optionally perform global augmentation for the training set if `GLOBAL_MULTIPLIER` is greater than 1.0 (default: `False`)
- `--balance-test` - Balance classes in created test set and optionally perform global augmentation for the test set if `GLOBAL_MULTIPLIER` is greater than 1.0 (default: `False`)
- `--output OUTPUT` - Path to an empty output directory (default: `None` - overwrite input directory)
- `--train-split TRAIN_SPLIT` - Train split ratio (default: `0.8`)
- `--seed SEED` - Random seed (default: `None`)
- `--label-col LABEL_COL` - Label column name (default: `'label'`)
- `--filename-col FILENAME_COL` - Filename column name (default: `'filename'`)
- `--global-multiplier GLOBAL_MULTIPLIER` - Global multiplier for the number of images in each class (default: `1.0`). This option can be used to increase the number of images in each class but is ignored if `--balance` is not used.
- `--pipeline-yaml` - Path to a custom Albumentations Compose pipeline serialized to YAML (default: `None` - use pipeline included in this script)

If you would like to use the `--pipeline-yaml` option, the following is a brief description of a custom pipeline and its serialization:

The pipeline has to be an instance of [`albumentations.core.composition.Compose`](https://albumentations.ai/docs/api_reference/core/composition/#albumentations.core.composition.Compose) and it must be serialized to a YAML file using [`albumentations.core.serialization.save`](https://albumentations.ai/docs/api_reference/core/serialization/#albumentations.core.serialization.save). The script will then internally be able to deserialize the pipeline using [`albumentations.core.serialization.load`](https://albumentations.ai/docs/api_reference/core/serialization/#albumentations.core.serialization.load).

Example of serializing a custom pipeline is included in the [`src`](./src) folder and is named [`custom_pipeline_example.py`](./src/custom_pipeline_example.py). Example of a serialized pipeline is included in the root folder and is named [`custom_pipeline_example.yml`](./custom_pipeline_example.yml).
