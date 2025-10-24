import os, sys
from config import DATASET_DIR, NUM_PARTITIONS
from load_dataset import load_data

x_train, y_train, x_test, y_test = load_data()

print('\nnum_samples, num_timestamps, num_features:', x_train[0].shape)
print('\nPartition 1:\n', x_train[0][0,:,0])
print('\nPartition 2:\n', x_train[1][0,:,0])
