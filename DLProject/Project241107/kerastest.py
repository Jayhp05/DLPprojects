import tensorflow as tf
from tensorflow import keras
import numpy as np
import tensorflowjs as tfjs

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1, input_shape=(1,)))
model.compile(optimizer='sgd', loss="mean_squared_error")
model.summary()
x = np.array([[1],[2],[3],[4]])
y = np.array([[1],[3],[5],[7]])
model.fit(x, y, epochs=100)
model.save('./testmodel.weights.h5')
model = tf.keras.models.load_model('./testmodel.weights.h5')
print(model.predict(np.array([[5]])))

tfjs.converters.save_keras_model(model, "./models/jstestmodel")