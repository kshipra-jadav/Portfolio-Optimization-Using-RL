import tensorflow as tf
from tensorflow.keras.utils import plot_model

def lstm_complex_model(input_shape, num_actions):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, input_shape=(input_shape, 1), activation='relu', return_sequences=True),
        tf.keras.layers.BatchNormalization(),  # Add Batch Normalization for better training stability
        tf.keras.layers.LSTM(32, activation='relu', return_sequences=True),
        tf.keras.layers.Dropout(0.2),  # Add Dropout for regularization
        tf.keras.layers.LSTM(16, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_actions, activation='linear')
    ])
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    return model

def lstm_simple_model(input_shape, num_actions):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(32, input_shape=(input_shape, 1), activation='relu'),
        tf.keras.layers.Dense(32, input_shape=(input_shape,), activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(num_actions, activation='linear')
    ])
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    return model

def gru_model(input_shape, num_actions):
    model = tf.keras.Sequential([
        tf.keras.layers.GRU(64, input_shape=(input_shape, 1), activation='relu', return_sequences=True),
        tf.keras.layers.BatchNormalization(),  # Add Batch Normalization for better training stability
        tf.keras.layers.GRU(32, activation='relu', return_sequences=True),
        tf.keras.layers.Dropout(0.2),  # Add Dropout for regularization
        tf.keras.layers.GRU(16, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_actions, activation='linear')
    ])
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    return model

if __name__ == '__main__':
    plot_model()