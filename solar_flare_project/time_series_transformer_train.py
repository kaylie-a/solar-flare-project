from config import PARTITIONS_DIR, MODELS_DIR, HISTORY_DIR, RESULTS_DIR, NUM_PARTITIONS, NUM_TIMESTEPS, NUM_FEATURES
from load_dataset import print_partition_info
from plot_graphs import plot_confusion_matrix

import numpy as np
import os
import json
import random
import matplotlib.pyplot as plt

import tensorflow.keras as tf
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


# TODO: Add time-series data shuffling?

# Reshape data for the transformer (batch_size, time_steps, features)
def reshape_data(x):
    return x.reshape((x.shape[0], x.shape[1], x.shape[2]))

# Encoder part of the Time-Series Transformer
def time_series_encoder(input_shape, num_classes=2):

    # Normalize input layer
    inputs = tf.Input(input_shape)
    x = tf.layers.LayerNormalization(epsilon=0.000001)(inputs)
    
    # Attention unit
    encoder_block = tf.layers.MultiHeadAttention(
        num_heads=8,
        key_dim=64,
        dropout=0.1
        )(x, x)
    encoder_block = tf.layers.Dropout(0.1)(encoder_block)

    # Residual block (add input to previous block)
    encoder_block = tf.layers.LayerNormalization(epsilon=0.000001)(encoder_block + x)
    encoder_block = tf.layers.ReLU()(encoder_block)
    
    # Feed-Forward network
    x = tf.layers.GlobalMaxPooling1D()(encoder_block)
    x = tf.layers.Dense(128, activation='relu')(x)
    x = tf.layers.Dropout(0.6)(x)

    # Add and normalize before output
    x = tf.layers.LayerNormalization(epsilon=0.000001)(x)
    output = tf.layers.Dense(num_classes, activation='softmax')(x)
    
    # Create and compile model
    encoder_model = tf.Model(inputs, output)
    encoder_model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=0.0001),
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

    print(f'\nValidation length: {len(y_val)}')
    print(f'Testing length: {len(y_test)}')

    print(f'\nValidation Class M: {len(np.where(y_val == 0)[0])}')
    print(f'Validation Class X: {len(np.where(y_val == 1)[0])}')
    print(f'Testing Class X: {len(np.where(y_test == 0)[0])}')
    print(f'Testing Class X: {len(np.where(y_test == 1)[0])}')

    return x_val, y_val, x_test, y_test


# =====================================================================

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)

iteration = 0
output_file = open(f"{RESULTS_DIR}/output_{iteration}.txt", "w")

# Training Partitions
for i in range(1):
    print(f'\nTraining Partition {i + 1}')
    print('=====================================================================')

    # Load training data
    current_train = np.load(f'{PARTITIONS_DIR}/train{i + 1}_test1.npz')
    x_train = current_train['x_train']
    y_train = current_train['y_train']
    
    # Preprocess and reshape training data
    x_train = reshape_data(x_train)

    # Create Time Series Transformer model
    model = time_series_encoder((NUM_TIMESTEPS, NUM_FEATURES))
    model.summary()

    # Test on different partitions
    for j in range(1):
        # Load testing partition
        current_test = np.load(f'{PARTITIONS_DIR}/train{i + 1}_test{j + 1}.npz')
        x_testing = current_test['x_test']
        y_testing = current_test['y_test']
        
        # Shuffle and split testing into two sets without reusing values
        x_val, y_val, x_test, y_test = split_val_test(x_testing, y_testing)

        # Print class distribution
        #print_partition_info(x_val, y_val, x_test, y_test)

        # Reshape validation and testing data
        x_val = reshape_data(x_val)
        x_test = reshape_data(x_test)

        # Train the model
        print(f'\nTesting Partition {j + 1}')
        
        history = model.fit(
            x_train, 
            y_train, 
            epochs=1,
            batch_size=32, 
            validation_data=(x_val, y_val),
            verbose=1
        )

        # Save the model and history
        model.save(f'{MODELS_DIR}/time_series_model_{i + 1}_{j + 1}.h5')
        history_output = f'{HISTORY_DIR}/history_{i + 1}_{j + 1}.json'
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

        # Save first confusion matrix
        if i + 1 == 1 and j + 1 == 1:
            plot_confusion_matrix(y_test, predictions, i + 1, j + 1, iteration)
        
        print(f'Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, TSS: {tss:.4f}')

        # Write to output file
        output_file.write(f'Pair {i + 1}:{j + 1} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, TSS: {tss:.4f}\n')
        