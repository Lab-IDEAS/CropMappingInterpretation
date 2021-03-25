# CropMappingInterpretation: An interpretation pipeline towards understanding multi-temporal deep learning approaches for crop mapping

This repository is the official implementation of the paper "An interpretation pipeline towards understanding multi-temporal deep learning approaches for crop mapping"

## Requirements

- torch
- numpy
- pandas
- scikit-learn
- jupyter
- matplotlib
- seaborn

The code has been tested in the following environment:
Ubuntu 16.04.4 LTS, Python 3.5.2, PyTorch 1.2.0

## Data

The preprocessed data (`.npy` files) for model training and evaluation is not directly provided here due to the large data volume. You can download raw Landsat Analysis Ready Data (ARD) from [EarthExplore](https://earthexplorer.usgs.gov/) and raw Cropland Data Layer (CDL) from [CropScape](https://nassgeodata.gmu.edu/CropScape/), then follow the code in the `preprocessing` folder to generate the `.npy` files. Specifically, run `preprocess_ARD.ipynb` at first, then `preprocess_CDL.ipynb`, and finally `category_binarization.ipynb`. The raw Landsat ARD and CDL data should be stored in a new `data` folder that has the following structure (specific downloaded file names may change):

```
data
├── Site_1
│   ├── ARD
│   │   ├── 2015
│   │   │   ├── LC08_CU_018007_20150424_20181206_C01_V01_PIXELQA.tif
│   │   │   ├── LC08_CU_018007_20150424_20181206_C01_V01_SRB2.tif
│   │   │   └── . . .
│   │   ├── . . .
│   │   └── 2018
│   └── CDL
│       ├── CDL_2015_clip_20190409130240_375669680.tif
│       ├── . . .
│       └── CDL_2018_clip_20190409125506_12566268.tif
├── Site_2
└── Site_3
```

The preprocessed data should be stored in the `preprocessing/out` folder that has the following structure:

```
preprocessing/out
├── Site_1
│   ├── x-2015.npy
│   ├── y-2015.npy
│   ├── x-corn_soybean-2015.npy
│   ├── y-corn_soybean-2015.npy
│   ├── . . .
│   ├── x-2018.npy
│   ├── y-2018.npy
│   ├── x-corn_soybean-2018.npy
│   └── y-corn_soybean-2018.npy
├── Site_2
└── Site_3
```

## Training and evaluation

- The PyTorch implementation of AtLSTM and Transformer models is located in the `models` folder. For random forest, the built-in implementation provided by scikit-learn is used.
- The `utils` folder contains some utilities that are used for data loading, normalization, training and evaluation.

The specific training and evaluation process can be executed by running the `.ipynb` files in the `experiments` folder.

## Understanding

The implementation of interpretation approaches is located in the `understanding` folder. Note that interpretation is based on the outputs of training and evaluation.
