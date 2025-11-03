from config import PARTITIONS_DIR, MODELS_DIR, NUM_PARTITIONS, NUM_FEATURES
import numpy as np
import os
import json
import tensorflow.keras as tf

# Reshape data for the transformer (batch_size, time_steps, features)
def reshape_data(x):
    return x.reshape((x.shape[0], x.shape[1], x.shape[2]))

# Decoder part of the Time-Series Transformer
def time_series_decoder(x, num_classes=1, timesteps=25):
    # Time-Series Decoder block
    decoder_block = tf.layers.MultiHeadAttention(
        num_heads=8,
        key_dim=64,
        dropout=0.1
        )(x, x)
    decoder_block = tf.layers.Dropout(0.1)(decoder_block)
    decoder_block = tf.layers.LayerNormalization(epsilon=0.000001)(decoder_block + x)

    # Make predictions for future 25 timesteps in each feature
    time_series_pred = tf.layers.Dense(NUM_FEATURES * timesteps, activation='linear')(time_series_pred)
    time_series_pred = reshape_data(time_series_pred)

    # Make label predictions
    label_pred = tf.layers.Dense(num_classes, activation='softmax')(time_series_pred)
    
    return time_series_pred, label_pred

# Combines encoder models to be passed to decoder
def combine_encoders(encoder_models, inputs):
    # Collect encoded outputs from each encoder model
    encoder_outputs = [encoder(inputs) for encoder in encoder_models]
    
    # Concatenate all encoder outputs along the feature dimension
    combined_output = tf.concat(encoder_outputs, axis=-1)
    return combined_output
