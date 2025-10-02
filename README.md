# Open ECG Digitizer

![Tests](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/actions/workflows/test.yml/badge.svg?branch=main) ![](https://img.shields.io/badge/%20style-google-3666d6.svg) [![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-31211/)

This repository provides a highly configurable tool for digitizing 12-lead ECGs, extracting raw time series data from scanned images or photographs (e.g., taken with a phone). It supports any subset of the 12 standard leads and is robust to perspective distortions and image quality variations.

---

## Features

- Extracts raw time series data from 12-lead ECG images
- Supports both scanned and photographed ECGs (with perspective correction)
- Works with any subset of leads
- Easily configurable via yaml config files

---

## Installation

**Requirements:** Python 3.12 or later.

**Note:** This setup has been tested on Ubuntu 24.04.2 and Debian 12 with CUDA.

1. Navigate to the project_source_code folder.
2. Create and activate a virtual environment: ```python3.12 -m venv venv && source venv/bin/activate```
3. Install dependencies ```python3 -m pip install -r requirements.txt```
4. Download the pre-trained weights: ```git lfs pull```

## Running inference
Run the inference on images in \<DATASET_FOLDER>\: ```python3 -m src.digitize --image_path <DATASET_FOLDER>```


## Benchmarking on the dataset
1. Run the inference first.
2. Run: ```python3 -m src.evaluate --image_path <DATASET_FOLDER>```


## Train on custom dataset
1. Change `data_path` for TRAIN, VAL and TEST in [src/config/unet.yml](src/config/unet.yml) to the locations of the custom dataset.
2. Run: ```python3 -m src.train```
