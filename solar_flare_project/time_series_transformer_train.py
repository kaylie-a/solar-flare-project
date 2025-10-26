from config import PARTITIONS_DIR, MODELS_DIR, HISTORY_DIR, NUM_PARTITIONS, NUM_TIMESTEPS, NUM_FEATURES
import numpy as np
import os
import json
import random
import matplotlib.pyplot as plt

import tensorflow.keras as tf
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns


# TODO: Add time-series data shuffling

# Reshape data for the transformer (batch_size, time_steps, features)
def reshape_data(x):
    return x.reshape((x.shape[0], x.shape[1], x.shape[2]))

# Encoder part of the Time-Series Transformer
def time_series_encoder(input_shape, num_classes=2):

    # Normalize input layer
    inputs = tf.Input(input_shape)
    x = tf.layers.LayerNormalization(epsilon=0.000001)(inputs)
    
    # Time-Series Encoder block 
    encoder_block = tf.layers.MultiHeadAttention(
        num_heads=8,
        key_dim=64,
        dropout=0.1
        )(x, x)
    encoder_block = tf.layers.Dropout(0.1)(encoder_block)
    # Residual block (add input to previous block)
    encoder_block = tf.layers.LayerNormalization(epsilon=0.000001)(encoder_block + x)
    
    # Feed-Forward network
    x = tf.layers.GlobalMaxPooling1D()(encoder_block)
    x = tf.layers.Dense(64, activation='relu')(x)
    x = tf.layers.Dropout(0.5)(x)
    output = tf.layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    encoder_model = tf.Model(inputs, output)
    encoder_model.compile(
        optimizer=tf.optimizers.Adam(),             # Backpropagation
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

# TODO: fix hardcoding
def save_confusion_matrix(y_true, predictions):

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
    plt.title('Pair 1:1')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('C:/GitHub/solar-flare-project/reports/figures/pair_1_1.png')
    plt.close()

# Split testing set into validation and testing
def split_val_test(x_testing, y_testing):

    indices = np.random.permutation(len(y_testing))
    
    # Split into validation and test sets
    split_indices = int(len(y_testing) * 0.5)

    val_indices = indices[:split_indices]   
    test_indices = indices[split_indices:]
    
    # Use the indices to get the corresponding data
    x_val = x_testing[val_indices, :, :]
    y_val = y_testing[val_indices]
    x_test = x_testing[test_indices, :, :]
    y_test = y_testing[test_indices]

    #print(f'y_val: {len(y_val)}')
    #print(f'y_test: {len(y_test)}')
    #print(f'x_val: {len(x_val)}')
    #print(f'x_test: {len(x_test)}')

    return x_val, y_val, x_test, y_test


# =====================================================================

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)

# Training Partitions
for i in range(NUM_PARTITIONS):
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
    for j in range(NUM_PARTITIONS):
        # Load testing partition
        current_test = np.load(f'{PARTITIONS_DIR}/train{i + 1}_test{j + 1}.npz')
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
            epochs=30,
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
            save_confusion_matrix(y_test, predictions)
        
        print(f'Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, TSS: {tss:.4f}')
