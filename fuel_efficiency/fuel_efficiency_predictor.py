import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
print(tf.__version__)

#Loading dataset
from pandas.core import indexing
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv('data/auto-mpg.csv', names=column_names,
                      na_values = "?", comment='\t',
                      sep=",",index_col=False,skiprows=1)

dataset = raw_dataset.copy()
dataset.tail()
# print(dataset.tail())
#Check count of missing values for each column
dataset.isna().sum()
# print(dataset.isna().sum())
#Check count of missing values for each column
dataset = dataset.dropna()
dataset
#one hot encoding:handling categorical columns
origin = dataset.pop('Origin')
dataset['USA'] = (origin == 1) * 1.0
dataset['Europe'] = (origin == 2) * 1.0
dataset['Japan'] = (origin == 3) * 1.0
dataset.tail()
# print(dataset.tail())

#Divide dataset into training and testing parts
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

#Seprate output field from other fields
#.pop() means "remove and return" a value from a collection like a list or a dictionary 
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

train_stats = train_dataset.describe()
train_stats = train_stats.transpose()

# print(train_stats)

#MODEL TRAINING

# bringing all columns to a common scale
def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)
# creating neural network

def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

model = build_model()
model.summary()
EPOCHS = 1000
history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS)

#Model Testing
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)
# print(loss,mae,mse)
test_predictions = model.predict(normed_test_data).flatten()
# print(test_predictions)
# print(test_labels)

#Model Conversion -saves the model in HDF5 format (.h5)
kearas_file = "automobile.keras"
tf.keras.models.save_model(model,kearas_file)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tfmodel = converter.convert()
open("automobile.tflite","wb").write(tfmodel)