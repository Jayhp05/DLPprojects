import tensorflow as tf
import tensorflowjs as tfjs
model = tf.keras.models.load_model('./mnist.h5')
tfjs.converters.save_keras_model(model, "./models/jstestmodel")