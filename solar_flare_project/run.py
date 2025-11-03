from config import HISTORY_DIR
from plot_graphs import plot_history
import json

training_pair = 1
testing_pair = 1
iteration = 0

# Load history
with open(f'{HISTORY_DIR}/history_{training_pair}_{testing_pair}.json', 'r') as f:
    history = json.load(f)

accuracy = history.get('accuracy', [])
val_accuracy = history.get('val_accuracy', [])
loss = history.get('loss', [])
val_loss = history.get('val_loss', [])

# Plot accuracy and loss
plot_history('Accuracy', accuracy, val_accuracy, training_pair, testing_pair, iteration)
plot_history('Loss', loss, val_loss, training_pair, testing_pair, iteration)
