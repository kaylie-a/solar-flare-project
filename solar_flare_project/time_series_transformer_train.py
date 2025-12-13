from config import PARTITIONS_DIR, MODELS_DIR, HISTORY_DIR, RESULTS_DIR, NUM_PARTITIONS, NUM_TIMESTEPS, NUM_FEATURES
from load_dataset import split_val_test, reshape_data
from plot_graphs import plot_confusion_matrix, plot_history

import numpy as np
import os
import json
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dropout, Dense, Add, GlobalMaxPooling1D, GlobalAveragePooling1D, Concatenate
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# =====================================================================

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)

iteration  = 12
num_layers = 3
dropout    = 0.1
num_epochs = 20

partitions_dir = PARTITIONS_DIR + '/processed'
output_file = open(f'{RESULTS_DIR}/output_{iteration}.txt', 'a')

# =====================================================================

# Encoder part of the Time-Series Transformer
def time_series_encoder(num_layers, dropout):
    # Normalize input layer
    input_shape = (NUM_TIMESTEPS, NUM_FEATURES)
    inputs = tfk.Input(input_shape)
    x = LayerNormalization(epsilon=0.000001)(inputs)
    
    # Positional Encoding
    pos = tf.linspace(0.0, 1.0, NUM_TIMESTEPS)[:, tf.newaxis]       # Timesteps
    i = tf.range(NUM_FEATURES, dtype=tf.float32)[tf.newaxis, :]     # Features

    # d_model = num_features
    # Even numbered i: sin(position / 10000^(2i/d_model))
    # Odd numbered i:  cos(position / 10000^(2i/d_model))   => i + 1
    frequency = 1 / tf.pow(10000, (2 * tf.floor(i / 2)) / NUM_FEATURES)
    positions = tf.where(
        tf.cast(i, tf.int32) % 2 == 0,
        tf.sin(pos * frequency),    # Even
        tf.cos(pos * frequency)     # Odd
    )

    # Add back to input
    x = x + positions

    # Encoder block
    for layer in range(num_layers):
        # Attention unit
        attention_unit = MultiHeadAttention(
            num_heads=4,
            key_dim=128,
            dropout=dropout,
            name=f'attention_unit_{layer}'
        )(x, x)

        # Residual block (add input to previous block)
        x = Add()([x, attention_unit])
        x = LayerNormalization(
            epsilon=0.000001,
            name=f'add_attention_{layer}'
        )(x)
    
        # Feed-Forward Network
        ffn_unit = Dense(
            128, 
            activation='relu', 
            kernel_regularizer=l2(0.0001),
            name=f'ffn_dense_{layer}'
        )(x)
        ffn_unit = Dropout(
            dropout,
            name=f'ffn_dropout_{layer}'
        )(ffn_unit)
        ffn_unit = Dense(
            NUM_FEATURES,
            kernel_regularizer=l2(0.0001),
            name=f'ffn_features_{layer}'
        )(ffn_unit)
        
        # Add input back again
        x = Add()([x, ffn_unit])
        x = LayerNormalization(
            epsilon=0.000001,
            name=f'add_fnn_normalize_{layer}'
        )(x)

    # Apply pooling to condense
    x = Concatenate()([
        GlobalAveragePooling1D()(x),
        GlobalMaxPooling1D()(x)
    ])

    # Add and normalize before output layer
    x = LayerNormalization(
        epsilon=0.000001,
        name=f'norm_output'
    )(x)
    output = Dense(
        2, 
        activation='softmax',
        name=f'out_softmax'
    )(x)
    
    encoder_model = Model(inputs, output)

    # Freeze layers after FFN
    for layer in encoder_model.layers[:-4]:
        layer.trainable = False

    encoder_model.compile(
        optimizer=tfk.optimizers.Adam(learning_rate=0.00001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return encoder_model


# Calculate TSS Score
def calc_tss(y_true, predictions):
    
    conf_mat = confusion_matrix(y_true, predictions)
    true_neg, false_pos, false_neg, true_pos = conf_mat.ravel()
    tss = (true_pos / (true_pos + false_neg)) - (false_pos / (false_pos + true_neg))
    return tss

# =====================================================================

def main():
    # Training Partitions
    for i in range(0, 1):
        print(f'\nTraining Partition {i + 1}')
        print('=====================================================================')

        # Load training data
        current_train = np.load(f'{partitions_dir}/train{i + 1}.npz')
        x_train = current_train['x_train']
        y_train = current_train['y_train']
        
        # Preprocess and reshape training data
        x_train = reshape_data(x_train)

        # Create Time Series Transformer model
        model = time_series_encoder(num_layers, dropout)
        model.summary()

        # Testing Partitions
        for j in range(NUM_PARTITIONS):
            # Load testing data from current partition
            current_test = np.load(f'{partitions_dir}/test{j + 1}.npz')
            x_testing = current_test['x_test']
            y_testing = current_test['y_test']
            
            # Shuffle and split testing into two sets without reusing values
            x_val, y_val, x_test, y_test = split_val_test(x_testing, y_testing)

            # Reshape validation and testing data
            x_val = reshape_data(x_val)
            x_test = reshape_data(x_test)

            # Train the model
            print(f'\nTesting Partition {j + 1}')
            
            history = model.fit(
                x_train, 
                y_train, 
                epochs=num_epochs,
                batch_size=32, 
                validation_data=(x_testing, y_testing),
                verbose=1
            )

            # Save the model and history
            model.save(f'{MODELS_DIR}/time_series_model_{i + 1}_{j + 1}_{iteration}.keras')
            history_output = f'{HISTORY_DIR}/history_{i + 1}_{j + 1}_{iteration}.json'
            with open(history_output, 'w') as file:
                json.dump(history.history, file)

            # Evaluate the model
            predictions = model.predict(x_test)
            predictions = np.argmax(predictions, axis=1)
            
            # Loss and accuracy
            loss, accuracy = model.evaluate(x_test, y_test)

            # Precision, Recall, F1-Score, TSS
            precision = precision_score(y_test, predictions, average='weighted')
            recall = recall_score(y_test, predictions, average='weighted')
            f1 = f1_score(y_test, predictions, average='weighted')
            tss = calc_tss(y_test, predictions)
            
            val_accuracy = history.history['val_accuracy']
            val_loss = history.history['val_loss']

            # Save confusion graphs
            plot_confusion_matrix(y_test, predictions, i + 1, j + 1, iteration)
            plot_history('Accuracy', accuracy, val_accuracy, i + 1, j + 1, iteration)
            plot_history('Loss', loss, val_loss, i + 1, j + 1, iteration)
            
            print(f'Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, TSS: {tss:.4f}')

            # Write to output file
            output_file.write(f'Pair {i + 1}:{j + 1} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, TSS: {tss:.4f}\n')

if __name__ == "__main__":
    main()
