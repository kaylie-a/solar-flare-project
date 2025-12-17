import numpy as np
import os
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

from config import NUM_PARTITIONS, PARTITIONS_DIR
from load_dataset import load_data

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

def augment_sample(x):
    augmentation = np.random.choice([0, 1])

    if augmentation == 0:
        noise = np.random.normal(0, 0.01, size=x.shape)
        return x + noise
    else:
        scale = np.random.normal(1.0, 0.1, size=(x.shape[0], 1, x.shape[2]))
        return x * scale

# =====================================================================

x_train_original, y_train_original, x_test, y_test = load_data()
os.makedirs(PARTITIONS_DIR, exist_ok=True)

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

    # Keep approx. 80/20 train/test split if there are too many samples
    if (num_samples > 40000):
        num_samples = 50000
    else:
        num_samples = len(m_class)

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

    random_state = np.random.default_rng(seed=42)
    new_sample_x = []
    new_sample_y = []

    # Number of class X samples needed to mach class M
    num_new_samples = num_samples - len(x_class_x)

    # Create new samples
    while len(new_sample_x) < num_new_samples:
        # Pick a random existing sample
        i = random_state.integers(0, len(x_class_x))
        current_sample = x_class_x[i : i + 1]

        # Augment sample
        new_sample = augment_sample(current_sample)

        # Add new class X sample
        new_sample_x.append(new_sample[0])
        new_sample_y.append(1.0)

    # Combine original and new class X samples
    if num_new_samples > 0:
        x_class_x = np.concatenate([x_class_x, np.stack(new_sample_x, axis=0)], axis=0)
        x_class_y = np.concatenate([x_class_y, np.array(new_sample_y)], axis=0)

    # Combine all samples
    x_combined = np.concatenate([m_class_x, x_class_x], axis=0)
    y_combined = np.concatenate([m_class_y, x_class_y], axis=0)

    # Shuffle examples
    rand = np.random.default_rng(seed=42)
    indices = np.arange(len(x_combined))
    rand.shuffle(indices)

    x_combined = x_combined[indices]
    y_combined = y_combined[indices]
    
    # Final x_train and y_train
    x_train.append(x_combined)
    y_train.append(y_combined)
    print(f'{len(x_train[partition])} total samples')

# Save processed data
for i in range(NUM_PARTITIONS):
    # Training partition
    output_dir = f'{PARTITIONS_DIR}/train{i + 1}.npz'
    np.savez(output_dir,
        x_train=x_train[i],
        y_train=y_train[i],
    )

    # Testing partition
    output_dir = f'{PARTITIONS_DIR}/test{i + 1}.npz'
    np.savez(output_dir,
        x_test=x_test[i],
        y_test=y_test[i],
    )
