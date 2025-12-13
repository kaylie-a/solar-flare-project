import numpy as np
import os
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

from config import NUM_PARTITIONS, PARTITIONS_DIR
from load_dataset import load_data

partitions_dir = PARTITIONS_DIR + '/processed'

x_train_original, y_train_original, x_test, y_test = load_data()
os.makedirs(partitions_dir, exist_ok=True)

# =======================
# Data pre-processing:
#   Class M always has more samples than Class X
#   => Undersample Class M
#   => Oversample  Class X
#
# Example:
#   Class M: 72238 samples  =>  50000 samples
#   Class X: 1254 samples   =>  50000 samples
# =======================

x_train = []
y_train = []

for partition in range(NUM_PARTITIONS):
    current_x = x_train_original[partition]
    current_y = y_train_original[partition]

    # Get training samples for each class
    m_class = np.where(current_y == 0.0)[0]
    x_class = np.where(current_y == 1.0)[0]
    
    # Find the nearest 10,000 number of samples
    num_samples = (len(m_class) // 10000) * 10000
    print(f'Partition {partition + 1}: {num_samples} samples', end='\t')

    # Downsample class M
    m_downsample = resample(
        m_class,
        replace=False,
        n_samples=num_samples,
        random_state=42
    )
    m_class_x = current_x[m_downsample]
    m_class_y = current_y[m_downsample]

    # Upsample class X - create new samples
    x_class_x = current_x[x_class]
    x_class_y = current_y[x_class]

    # Combine samples
    x_combined = np.concatenate([m_class_x, x_class_x], axis=0)
    y_combined = np.concatenate([m_class_y, x_class_y], axis=0)

    # Flatten data
    samples, timesteps, features = x_combined.shape
    flatten_x = x_combined.reshape(samples, (timesteps * features))

    # Create new samples with SMOTE
    smote = SMOTE(
        sampling_strategy={1: num_samples},
        random_state=42
    )

    x_upsample_x, x_upsample_y = smote.fit_resample(flatten_x, y_combined)
    x_upsample = x_upsample_x.reshape(-1, timesteps, features)

    # Shuffling
    rand = np.random.default_rng(seed=42)
    indices = np.arange(len(x_upsample))
    rand.shuffle(indices)

    x_upsample = x_upsample[indices]
    x_upsample_y = x_upsample_y[indices]
    
    # Final x_train and y_train
    x_train.append(x_upsample)
    y_train.append(x_upsample_y)
    print(f'{len(x_train[partition])} new samples')

# Save processed data
for i in range(NUM_PARTITIONS):
    # Training partition
    output_dir = f'{partitions_dir}/train{i + 1}.npz'
    np.savez(output_dir,
        x_train=x_train[i],
        y_train=y_train[i],
    )

    # Testing partition
    output_dir = f'{partitions_dir}/test{i + 1}.npz'
    np.savez(output_dir,
        x_test=x_test[i],
        y_test=y_test[i],
    )
