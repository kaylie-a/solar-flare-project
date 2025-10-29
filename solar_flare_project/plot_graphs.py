from config import FIGURES_DIR, NUM_TIMESTEPS, FEATURE_NAMES
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

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
    
    num_features = len(list_features)
    timesteps = np.arange(NUM_TIMESTEPS)

    plt.figure(figsize=(7, 5))  # 10, 15 for full

    for i in range(num_features):
        plt.plot(timesteps, sample[example_index,:,list_features[i]], label=FEATURE_NAMES[list_features[i]])

    plt.xlabel('Timesteps')
    plt.ylabel('Values')
    plt.title(title)
    plt.legend()
    
    output_path = f'{FIGURES_DIR}/timeseries_example_small_{example_index}.png'
    plt.savefig(output_path)
    plt.close()
