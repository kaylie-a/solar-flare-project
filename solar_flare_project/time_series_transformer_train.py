from config import PARTITIONS_DIR, NUM_PARTITIONS
import numpy as np

for i in range(NUM_PARTITIONS):
    for j in range(NUM_PARTITIONS):
        current_pair = np.load(f'{PARTITIONS_DIR}/train{i + 1}_test{j + 1}.npz')
        x_train = current_pair['x_train']
        y_train = current_pair['y_train']
        x_test = current_pair['x_test']
        y_test = current_pair['y_test']

        # Perform training
