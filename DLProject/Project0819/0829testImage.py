from PIL import Image
import os
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
 
data_dir = "/home/aisw/DLProject/Project0819/images"
files = os.listdir(data_dir)
 
images = []
for file in files:
    path = os.path.join(data_dir, file)
    images.append(np.array(Image.open(path)))
 
resized_images = np.array(np.zeros((len(images), 224, 224, 3)))
for i in range(len(images)):
    resized_images[i] = tf.image.resize(images[i], [224, 224])
preprocesse_images = tf.keras.applications.mobilenet_v2.preprocess_input(resized_images)
 
mobilenet_imagenet_model = tf.keras.applications.MobileNetV2(weights="imagenet")
y_pred = mobilenet_imagenet_model.predict(preprocesse_images)
topK = 1
y_pred_top = tf.keras.applications.mobilenet_v2.decode_predictions(y_pred, top=topK)
 
for i in range(len(images)):
    plt.imshow(images[i])
    plt.show()
    for k in range(topK):
        print(f'{y_pred_top[i][k][1]} ({round(y_pred_top[i][k][2] * 100, 1)})%)')
