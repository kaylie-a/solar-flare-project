from config import DATASET_DIR, NUM_PARTITIONS
import pickle as pkl
import numpy as np

training_dir = DATASET_DIR + '/train'
testing_dir = DATASET_DIR + '/test'

############################################
#   training_array[a][b,:,c]
#       a: Partition
#           Total 5 partitions (0-4)
#       b: Sample number (time sequence)
#           60 timestamps
#       c: Feature index
#           Total 24 features:
#           ['R_VALUE', 'TOTUSJH', 'TOTBSQ', 'TOTPOT', 'TOTUSJZ', 'ABSNJZH', 'SAVNCPP', 'USFLUX',
#            'TOTFZ', 'MEANPOT', 'EPSX', 'EPSY', 'EPSZ', 'MEANSHR', 'SHRGT45', 'MEANGAM', 'MEANGBT', 'MEANGBZ', 'MEANGBH',
#            'MEANJZH', 'TOTFY', 'MEANJZD', 'MEANALP', 'TOTFX']
#   testing_array[a][b]
#       a: Partition (1-5)
#       b: Sample Label (0 or 1)
#
#   Example (training data):
#       x_train[2][1,:,1]
#           - Partition 3
#           - Second sample time-series data ...
#           - for feature 'TOTUSJH'
#
#   Example (testing data):
#       y_test[0][2]
#           - Partition 1
#           - Third sample label (0 or 1)
############################################

# Load data from each partition for training and testing
def load_data(training_dir=training_dir, testing_dir=testing_dir):

    x_train, y_train, x_test, y_test = [], [], [], []

    for i in range(NUM_PARTITIONS):
        with open(f'{training_dir}/Partition{i + 1}_LSBZM-Norm_FPCKNN-impute.pkl', 'rb') as file:
            load_x = pkl.load(file)
            x_train.append(load_x)

        with open(f'{training_dir}/Partition{i + 1}_Labels_LSBZM-Norm_FPCKNN-impute.pkl', 'rb') as file:
            load_y = pkl.load(file)
            y_train.append(load_y)

        with open(f'{testing_dir}/Partition{i + 1}_RUS-Tomek-TimeGAN_LSBZM-Norm_WithoutC_FPCKNN-impute.pkl', 'rb') as file:
            load_x = pkl.load(file)
            x_test.append(load_x)

        with open(f'{testing_dir}/Partition{i + 1}_Labels_RUS-Tomek-TimeGAN_LSBZM-Norm_WithoutC_FPCKNN-impute.pkl', 'rb') as file:
            load_y = pkl.load(file)
            y_test.append(load_y)

    return x_train, y_train, x_test, y_test

# Split testing set into validation and testing
def split_val_test(x_testing, y_testing):

    # Shuffle indices from each class
    M_indices = np.where(y_testing == 0)[0]
    X_indices = np.where(y_testing == 1)[0]
    np.random.shuffle(M_indices)
    np.random.shuffle(X_indices)
    
    # Split indices into validation and testing
    split_M = len(M_indices) // 2
    split_X = len(X_indices) // 2

    # Validation set
    x_val_M = M_indices[:split_M]
    x_val_X = X_indices[:split_X]
    val_indices = np.concatenate([x_val_M, x_val_X])

    # Testing set
    y_val_M = M_indices[split_M:]
    y_val_X = X_indices[split_X:]
    test_indices = np.concatenate([y_val_M, y_val_X])
    
    # Get final validation and testing sets
    x_val = x_testing[val_indices, :, :]
    y_val = y_testing[val_indices]
    x_test = x_testing[test_indices, :, :]
    y_test = y_testing[test_indices]

    return x_val, y_val, x_test, y_test

# Reshape data for the transformer (batch_size, time_steps, features)
def reshape_data(x):
    return x.reshape((x.shape[0], x.shape[1], x.shape[2]))
