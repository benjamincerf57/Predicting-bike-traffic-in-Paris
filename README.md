### Challenge description

This challenge focuses on predicting bicycle traffic volumes using data from automated bike counters installed throughout Paris. The goal is to develop a machine learning model that can accurately forecast the number of cyclists passing through different counting stations. This project is a part of the X-HEC Data Science master's program (first year). The contributors are CERF Benjamin and DE CIAN Charles. You can find enclosed various folders containing the code, the data and our final report.

# Starting kit on the bike counters dataset

![GH Actions](https://github.com/ramp-kits/bike_counters/actions/workflows/main.yml/badge.svg)

## Getting started

### Download the data,

Download the data files,
 - [train.parquet](https://github.com/ramp-kits/bike_counters/releases/download/v0.1.0/train.parquet)
 - [test.parquet](https://github.com/ramp-kits/bike_counters/releases/download/v0.1.0/test.parquet)

and put them into into the data folder.

Note that the `test.parquet` file is different from the actual `final_test.parquet` used for the evaluation on Kaggle. This file is just here for convenience.

### Install

To run the notebook you will need the dependencies listed
in `requirements.txt`. 

It is recommended to create a new virtual environement for this project. For instance, with conda,
```bash
conda create -n bikes-count python=3.10
conda activate bikes-count
```

You can install the dependencies with the following command-line:

```bash
pip install -U -r requirements.txt
```