import tensorflow as tf

# To ignore the warning which caused the model not to be saved
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

fashion_mnist = tf.keras.datasets.fashion_mnist
(image_train, label_train), (image_test, label_test) = fashion_mnist.load_data()


# # Normalize the data to 0-1 range
image_train = image_train / 255.0
image_test = image_test / 255.0


model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(256, activation=tf.nn.sigmoid),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(image_train, label_train, epochs=10)

model_loss, model_acc = model.evaluate(image_test, label_test)
print(f"Loss: {model_loss}, Accuracy: {model_acc}")

print(f"Model Summary")
model.summary()


model.save('fashion_mnist.model')
# new_model = tf.keras.models.load_model('fashion.model')
