import tensorflow as tf

mobilenet_imagenet_model = tf.keras.applications.MobileNetV2(weights="imagenet")

converter = tf.lite.TFLiteConverter.from_keras_model(mobilenet_imagenet_model)
tflite_model = converter.convert()

with open('./mobilenet_imagenet_model.tflite', 'wb') as f:
    f.write(tflite_model)