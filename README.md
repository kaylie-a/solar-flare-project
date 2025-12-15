# SOLAR FLARE CLASSIFICATION

Using a cleaned dataset with solar flare data, we analyzed the performance of two Transformer architectures in predicting solar flare levels across various time periods. Between a baseline time-series and PatchTST encoder, we compared how well each were able to analyze numerical data.

## Project Organization

```
├── LICENSE
│
├── README.md
│
├── requirements.txt                   <- Install required libraries for environment
│
├── pathTST                            <- Code for running PatchTST
│   └── 0_5_JL_patchTST.ipynb
│
├── base_timeseries                    <- Code for running baseline Time-Series
│   ├── base_time_series.py            <- Run training
│   ├── config.py                      <- Configure local folder paths
│   ├── load_dataset.py                <- Functions for working with the dataset
│   ├── plot_graphs.py                 <- Functions for saving graphs to images
│   └── preprocessed_partitions.py     <- Resamples unbalanced dataset
│
├── data\raw                           <- Original dataset
│
├── data_partitions                    <- Partition data after preprocessing (total of 10), generated from base_timeseries\preprocessed_partition.py
│   ├── test1.npz                      <- Testing partition 1
│   ├── train1.npz                     <- Training partition 1
│   └── . . .
│
├── reports                            <- Saved outputs from training runs
│   ├── figures                        <- Graph images
│   └── results                        <- .txt files with evaluation metrics
│
├── training                           <- Saved outputs from training runs
│   ├── history                        <- Accuracy and loss data
│   └── models                         <- .keras models
│
```

--------

## Source
The dataset used for this project is the SWAN-SF solar flare dataset.  
Original repository: [https://github.com/samresume/Cleaned-SWANSF-Dataset](https://github.com/samresume/Cleaned-SWANSF-Dataset)

## Local Setup
Download the dataset and place it in: data\raw
