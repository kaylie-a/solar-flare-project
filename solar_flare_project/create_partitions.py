import numpy as np
import os
from config import DATASET_DIR, NUM_PARTITIONS, NUM_FEATURES
from load_dataset import load_data, plot_timeseries_data

x_train, y_train, x_test, y_test = load_data()

#list_features = [1, 2, 3, 4, 5]
#list_features = np.arange(NUM_FEATURES)
#plot_timeseries_data(x_train[0], list_features, 'Class M Time-Series Data - Partition 1', 0)
#plot_timeseries_data(x_train[0], list_features, 'Class X Time-Series Data - Partition 1', 52)

partitions_dir = 'C:/GitHub/solar-flare-project/data_partitions'
os.makedirs(partitions_dir, exist_ok=True)

# Save partition pairs with training:testing
# Training partitions
for i in range(NUM_PARTITIONS):
    # Testing partitions
    for j in range(NUM_PARTITIONS):
        # Create partition pair
        x_train_pair = x_train[i]
        y_train_pair = y_train[i]
        x_test_pair = x_test[j]
        y_test_pair = y_test[j]

        # Save pair to directory
        output_dir = f'{partitions_dir}/train{i + 1}_test{j + 1}.npz'
        np.savez(output_dir,
                 x_train=x_train_pair,
                 y_train=y_train_pair,
                 x_test=x_test_pair,
                 y_test=y_test_pair
                 )

