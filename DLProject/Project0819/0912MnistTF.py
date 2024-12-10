import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, xtest = x_train / 255.0, x_test / 255.0

x_train_4d = x_train.reshape(-1, 28, 28, 1)
x_test_4d = x_test.reshape(-1, 28, 28, 1)

resized_x_train = tf.image.resize(x_train_4d, [28, 28])
resized_x_test = tf.image.resize(x_test_4d, [28, 28])

resnet_model = tf.keras.applications.ResNet50V2(
    input_shape=(32,32,1),
    classes = 1000,
    weights = 'imagenet'
    )

resnet_model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])

resnet_model.summary()

checkPointCallback = tf.keras.callbacks.ModelCheckpoint(
    "./mnist.keras", monitor='val_accuracy', verbose=1, save_weights_only=False ,save_best_only=True, mode='max')
 
resnet_model.fit(resized_x_train, y_train,
                 validation_data=(resized_x_test, y_test),
                 callbacks=[checkPointCallback],
                 epochs=5)

resnet_model = tf.keras.models.load_model('./mnist.keras')
test_loss, test_acc = resnet_model.evaluate(resized_x_test, y_test)
print(test_loss, "  ,   ", test_acc)

converter = tf.lite.TFLiteConverter.from_keras_model(resnet_model)
tflite_model = converter.convert()
with open('./mnist_model.tflite', 'wb') as f:
    f.write(tflite_model)