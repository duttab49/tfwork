import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from os import getcwd

def get_data(filename):
    with open(filename) as training_file:
        # Your code starts here
        data = np.genfromtxt(training_file, delimiter=',', skip_header=1, dtype=np.uint8)
        labels = data[:, 0]
        images = data[:, 1:]
        images = np.reshape(images, (data.shape[0], 28, 28))
        data = None
        # Your code ends here
    return images, labels

path_sign_mnist_train = f"{getcwd()}/../tmp2/sign_mnist_train.csv"
path_sign_mnist_test = f"{getcwd()}/../tmp2/sign_mnist_test.csv"
training_images, training_labels = get_data(path_sign_mnist_train)
testing_images, testing_labels = get_data(path_sign_mnist_test)

# Keep these
print(training_images.shape)
print(training_labels.shape)
print(testing_images.shape)
print(testing_labels.shape)

training_images = np.expand_dims(training_images, axis=-1)
testing_images = np.expand_dims(testing_images, axis=-1)

# Create an ImageDataGenerator and do Image Augmentation
train_datagen = ImageDataGenerator(
            rescale = 1./255,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.10,
            horizontal_flip=False,
            fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(
    rescale = 1./255)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPool2D(2, 2, padding='same'),

    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation="relu"),
    tf.keras.layers.MaxPool2D(2, 2, padding='same'),

    # tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation="relu"),
    # tf.keras.layers.MaxPool2D(2,2, padding='same'),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(units=1024, activation="relu"),
    tf.keras.layers.Dense(units=256, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=25, activation="softmax")
    # tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
    # tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    # tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    # tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    # tf.keras.layers.Flatten(),
    # tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    # tf.keras.layers.Dense(512, activation='relu'),
    # tf.keras.layers.Dense(3, activation='softmax')
])

# Compile Model.
model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the Model
train_flow = train_datagen.flow(training_images, training_labels, batch_size=32)
val_flow = validation_datagen.flow(testing_images, testing_labels, batch_size=32)

history = model.fit_generator(train_flow,
                              validation_data=val_flow,
                              epochs=2)

model.evaluate(testing_images, testing_labels, verbose=0)


import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

