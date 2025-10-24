import pickle as pkl
import os, sys

# Add one directory above to path
from config import DATASET_DIR, NUM_PARTITIONS

training_dir = DATASET_DIR + '/train'
testing_dir = DATASET_DIR + '/test'
x_train, y_train, x_test, y_test = [], [], [], []

############################################
#   training_array[a][b,:,c]
#       a: Partition (1-5)
#       b: Sample number (time sequence)
#       c: Feature index
#           ['R_VALUE', 'TOTUSJH', 'TOTBSQ', 'TOTPOT', 'TOTUSJZ', 'ABSNJZH', 'SAVNCPP', 'USFLUX',
#            'TOTFZ', 'MEANPOT', 'EPSX', 'EPSY', 'EPSZ', 'MEANSHR', 'SHRGT45', 'MEANGAM', 'MEANGBT', 'MEANGBZ', 'MEANGBH',
#            'MEANJZH', 'TOTFY', 'MEANJZD', 'MEANALP', 'TOTFX']
#   testing_array[a][b]
#       a: Partition (1-5)
#       b: Sample Label
#
#   Example (training data):
#       x_train[2][1,:,1]
#           - Partition 3
#           - Second sample
#           - Feature 'TOTUSJH'
#
#   Example (testing data):
#       y_test[0][2]
#           - Partition 1
#           - Third sample label (0 or 1)
############################################

# Load data from each partition for training and testing
def load_data(training_dir=training_dir, testing_dir=testing_dir):
    for i in range(NUM_PARTITIONS):
        with open(f'{training_dir}/Partition{i + 1}_RUS-Tomek-TimeGAN_LSBZM-Norm_WithoutC_FPCKNN-impute.pkl', 'rb') as file:
            load_x = pkl.load(file)
            x_train.append(load_x)

        with open(f'{training_dir}/Partition{i + 1}_Labels_RUS-Tomek-TimeGAN_LSBZM-Norm_WithoutC_FPCKNN-impute.pkl', 'rb') as file:
            load_y = pkl.load(file)
            y_train.append(load_y)

        with open(f'{testing_dir}/Partition{i + 1}_LSBZM-Norm_FPCKNN-impute.pkl', 'rb') as file:
            load_x = pkl.load(file)
            x_test.append(load_x)

        with open(f'{testing_dir}/Partition{i + 1}_Labels_LSBZM-Norm_FPCKNN-impute.pkl', 'rb') as file:
            load_y = pkl.load(file)
            y_test.append(load_y)

    return x_train, y_train, x_test, y_test
