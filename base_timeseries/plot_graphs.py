from config import FIGURES_DIR, NUM_TIMESTEPS, NUM_FEATURES, FEATURE_NAMES
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Plot accuracy or loss during training
def plot_history(metric, training, validation, training_pair, testing_pair, iteration):
    plt.figure(figsize=(7, 5))
    plt.plot(training, label=f'Training {metric}')
    plt.plot(validation, label=f'Validation {metric}')
    plt.title(f'Pair {training_pair}-{testing_pair} Training {metric}')
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.legend()
    plt.savefig(f'{FIGURES_DIR}/{metric.lower()}_{training_pair}_{testing_pair}_{iteration}.png')
    plt.close()

def plot_confusion_matrix(y_true, predictions, training_pair, testing_pair, iteration):

    conf_mat = confusion_matrix(y_true, predictions)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        conf_mat, 
        annot=True, 
        fmt='d',
        cmap='Blues', 
        cbar=True, 
        xticklabels=['Class M', 'Class X'], 
        yticklabels=['Class M', 'Class X']
        )
    plt.title(f'Pair {training_pair}:{testing_pair}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'{FIGURES_DIR}/pair_{training_pair}_{testing_pair}_{iteration}.png')
    plt.close()

# Plot time-series data for different features of one sample
def plot_timeseries_data(sample, list_features, title, example_index=0):
    
    timesteps = np.arange(NUM_TIMESTEPS)

    plt.figure(figsize=(7, 5))  # 10, 15 for full

    for i in range(NUM_FEATURES):
        plt.plot(timesteps, sample[example_index,:,list_features[i]], label=FEATURE_NAMES[list_features[i]])

    plt.xlabel('Timesteps')
    plt.ylabel('Values')
    plt.title(title)
    plt.legend()
    
    output_path = f'{FIGURES_DIR}/timeseries_example_small_{example_index}.png'
    plt.savefig(output_path)
    plt.close()

# Compare feature data between two partitions
def plot_feature_example(x_train, x_test, train_partition, test_partition, feature, num_examples):
    feature_training = []
    feature_testing = []

    for i in range(num_examples):
        for j in range(NUM_TIMESTEPS):
            feature_training.append(x_train[train_partition][i, :, feature][j])
            feature_testing.append(x_test[test_partition][i, :, feature][j])

    plt.figure(figsize=(20, 10))
    plt.plot(feature_training, label=f'Train {train_partition + 1} - {FEATURE_NAMES[feature]}')
    plt.plot(feature_testing, label=f'Test {test_partition + 1} - {FEATURE_NAMES[feature]}')
    plt.xlabel('Timesteps')
    plt.ylabel('Values')
    plt.title(f'Example')
    plt.legend()

    output_path = f'{FIGURES_DIR}/{FEATURE_NAMES[feature]}_train{train_partition + 1}_test{test_partition + 1}.png'
    plt.savefig(output_path)
    plt.close()
