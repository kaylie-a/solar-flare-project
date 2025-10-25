import numpy as np
import os
from config import NUM_PARTITIONS, PARTITIONS_DIR
from load_dataset import load_data

x_train, y_train, x_test, y_test = load_data()
os.makedirs(PARTITIONS_DIR, exist_ok=True)

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
        output_dir = f'{PARTITIONS_DIR}/train{i + 1}_test{j + 1}.npz'
        np.savez(output_dir,
                 x_train=x_train_pair,
                 y_train=y_train_pair,
                 x_test=x_test_pair,
                 y_test=y_test_pair
                 )
