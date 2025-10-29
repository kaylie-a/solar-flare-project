from config import FIGURES_DIR
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
