import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from utils.preprocessor import load_and_preprocess_data
from utils.exporter import save_model

# Load data
train_data, train_labels, test_data, test_labels, _ = load_and_preprocess_data('data/auto-mpg.csv')

# Build model
def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[train_data.shape[1]]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(0.001),
        loss='mse',
        metrics=['mae', 'mse']
    )
    return model

model = build_model()
model.fit(train_data, train_labels, epochs=1000)

# Evaluate
loss, mae, mse = model.evaluate(test_data, test_labels, verbose=0)
print(f"Test MAE: {mae:.4f}, MSE: {mse:.4f}")

# Export model
save_model(model)








