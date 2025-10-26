from config import PARTITIONS_DIR, MODELS_DIR, HISTORY_DIR, NUM_PARTITIONS, NUM_TIMESTEPS, NUM_FEATURES
import numpy as np
import os
import json
import tensorflow.keras as tf

# Add shuffling later

# Reshape data for the transformer (batch_size, time_steps, features)
def reshape_data(x):
    return x.reshape((x.shape[0], x.shape[1], x.shape[2]))

def time_series_transformer_model(input_shape, num_classes=2):
    inputs = tf.Input(input_shape)
    
    # Normalize the time-series inputs
    x = tf.layers.LayerNormalization(epsilon=0.000001)(inputs)
    
    # Time-Series Transformer block 
    transformer_block = tf.layers.MultiHeadAttention(
        num_heads=8, 
        key_dim=64, 
        dropout=0.1
        )(x, x)
    transformer_block = tf.layers.Dropout(0.1)(transformer_block)
    # Residual block (add input to previous block)
    transformer_block = tf.layers.LayerNormalization(epsilon=0.000001)(transformer_block + x)
    
    # Feed-Forward network
    x = tf.layers.GlobalAveragePooling1D()(transformer_block)
    x = tf.layers.Dense(64, activation='relu')(x)
    x = tf.layers.Dropout(0.5)(x)
    output = tf.layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = tf.Model(inputs, output)
    model.compile(
        optimizer=tf.optimizers.Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
        )
    
    return model

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)

# Save models
for i in range(1):
    # Load training data
    current_train = np.load(f'{PARTITIONS_DIR}/train{i + 1}_test1.npz')
    x_train = current_train['x_train']
    y_train = current_train['y_train']
    
    # Preprocess and reshape training data
    x_train = reshape_data(x_train)

    # Create Time Series Transformer model
    model = time_series_transformer_model((NUM_TIMESTEPS, NUM_FEATURES))
    
    # Train the model
    history = model.fit(
        x_train, 
        y_train, 
        epochs=5,
        batch_size=32, 
        verbose=1
    )

    # Save the model and history
    model.save(f'{MODELS_DIR}/time_series_model_{i + 1}.h5')
    history_output = f'{HISTORY_DIR}/history_{i + 1}.json'
    with open(history_output, 'w') as file:
        json.dump(history.history, file)
    
    # TODO: move to separate file later
    # Test on different partitions
    for j in range(1):
        # Load testing partition
        current_test = np.load(f'{PARTITIONS_DIR}/train{i + 1}_test{j + 1}.npz')
        x_test = current_test['x_test']
        y_test = current_test['y_test']
        
        # Preprocess and reshape testing data
        x_test = reshape_data(x_test)

        # Evaluate the model
        loss, accuracy = model.evaluate(x_test, y_test)
        print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
