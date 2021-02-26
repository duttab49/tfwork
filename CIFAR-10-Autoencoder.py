# CIFAR-10 Autoencoder
try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass

import tensorflow as tf
import tensorflow_datasets as tfds

from keras.models import Sequential

# preprocessing function
def map_image(image, label):
  image = tf.cast(image, dtype=tf.float32)
  image = image / 255.0

  return image, image # dataset label is not used. replaced with the same image input.

# parameters
BATCH_SIZE = 128
SHUFFLE_BUFFER_SIZE = 1024

# use tfds.load() to fetch the 'train' split of CIFAR-10
train_dataset = tfds.load('cifar10', as_supervised=True, split="train")

# preprocess the dataset with the `map_image()` function above
train_dataset = train_dataset.map(map_image)

# shuffle and batch the dataset
train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

# use tfds.load() to fetch the 'test' split of CIFAR-10
test_dataset = tfds.load('cifar10', as_supervised=True, split="test")

# preprocess the dataset with the `map_image()` function above
test_dataset = test_dataset.map(map_image)

# batch the dataset
test_dataset = test_dataset.batch(BATCH_SIZE)

# Build the Model
# suggested layers to use.
from keras.layers import Conv2D, UpSampling2D, MaxPooling2D

# use the Sequential API
model = Sequential()

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same', input_shape=(32,32,3)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

# Defines bottleneck
model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same'))

# Defines the decoder path to upsample back to the original image size.
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.UpSampling2D(size=(2,2)))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.UpSampling2D(size=(2,2)))
model.add(tf.keras.layers.Conv2D(filters=3, kernel_size=(3,3), activation='sigmoid', padding='same'))

model.summary()

#Model: "sequential_2"
#_________________________________________________________________
#Layer (type)                 Output Shape              Param #   
#=================================================================
#conv2d_47 (Conv2D)           (None, 32, 32, 64)        1792      
#_________________________________________________________________
#max_pooling2d_14 (MaxPooling (None, 16, 16, 64)        0         
#_________________________________________________________________
#conv2d_48 (Conv2D)           (None, 16, 16, 128)       73856     
#_________________________________________________________________
#max_pooling2d_15 (MaxPooling (None, 8, 8, 128)         0         
#_________________________________________________________________
#conv2d_49 (Conv2D)           (None, 8, 8, 256)         295168    
#_________________________________________________________________
#conv2d_50 (Conv2D)           (None, 8, 8, 128)         295040    
#_________________________________________________________________
#up_sampling2d_14 (UpSampling (None, 16, 16, 128)       0         
#_________________________________________________________________
#conv2d_51 (Conv2D)           (None, 16, 16, 64)        73792     
#_________________________________________________________________
#up_sampling2d_15 (UpSampling (None, 32, 32, 64)        0         
#_________________________________________________________________
#conv2d_52 (Conv2D)           (None, 32, 32, 3)         1731      
#=================================================================
#Total params: 741,379
#Trainable params: 741,379
#Non-trainable params: 0
#_________________________________________________________________

# Configure training parameters
model.compile(optimizer='adam', metrics=['accuracy'], loss='mean_squared_error')

# Training
train_steps = len(train_dataset) // BATCH_SIZE 
val_steps = len(test_dataset) // BATCH_SIZE

model.fit(train_dataset, steps_per_epoch=train_steps, validation_data=test_dataset, validation_steps=val_steps, epochs=40)

#Epoch 1/40
#3/3 [==============================] - 1s 67ms/step - loss: 0.0648 - accuracy: 0.4717
#Epoch 2/40
#3/3 [==============================] - 0s 54ms/step - loss: 0.0605 - accuracy: 0.4851
#Epoch 3/40
#3/3 [==============================] - 0s 50ms/step - loss: 0.0525 - accuracy: 0.4944
#Epoch 4/40
#3/3 [==============================] - 0s 49ms/step - loss: 0.0487 - accuracy: 0.4719
#Epoch 5/40
#3/3 [==============================] - 0s 52ms/step - loss: 0.0425 - accuracy: 0.3854
#Epoch 6/40
#3/3 [==============================] - 0s 51ms/step - loss: 0.0349 - accuracy: 0.3698
#Epoch 7/40
#3/3 [==============================] - 0s 50ms/step - loss: 0.0299 - accuracy: 0.5275
#Epoch 8/40
#3/3 [==============================] - 0s 51ms/step - loss: 0.0256 - accuracy: 0.5347
#Epoch 9/40
#3/3 [==============================] - 0s 53ms/step - loss: 0.0229 - accuracy: 0.5406
#Epoch 10/40
#3/3 [==============================] - 0s 47ms/step - loss: 0.0206 - accuracy: 0.5913
#Epoch 11/40
#3/3 [==============================] - 0s 50ms/step - loss: 0.0193 - accuracy: 0.6163
#Epoch 12/40
#3/3 [==============================] - 0s 50ms/step - loss: 0.0176 - accuracy: 0.5995
#Epoch 13/40
#3/3 [==============================] - 0s 54ms/step - loss: 0.0163 - accuracy: 0.6127
#Epoch 14/40
#3/3 [==============================] - 0s 52ms/step - loss: 0.0147 - accuracy: 0.6250
#Epoch 15/40
#3/3 [==============================] - 0s 49ms/step - loss: 0.0140 - accuracy: 0.6545
#Epoch 16/40
#3/3 [==============================] - 0s 53ms/step - loss: 0.0139 - accuracy: 0.6435
#Epoch 17/40
#3/3 [==============================] - 0s 50ms/step - loss: 0.0128 - accuracy: 0.6392
#Epoch 18/40
#3/3 [==============================] - 0s 51ms/step - loss: 0.0132 - accuracy: 0.6354
#Epoch 19/40
#3/3 [==============================] - 0s 48ms/step - loss: 0.0118 - accuracy: 0.6245
#Epoch 20/40
#3/3 [==============================] - 0s 50ms/step - loss: 0.0113 - accuracy: 0.6560
#Epoch 21/40
#3/3 [==============================] - 0s 49ms/step - loss: 0.0111 - accuracy: 0.6534
#Epoch 22/40
#3/3 [==============================] - 0s 50ms/step - loss: 0.0110 - accuracy: 0.6653
#Epoch 23/40
#3/3 [==============================] - 0s 51ms/step - loss: 0.0100 - accuracy: 0.6684
#Epoch 24/40
#3/3 [==============================] - 0s 52ms/step - loss: 0.0095 - accuracy: 0.6615
#Epoch 25/40
#3/3 [==============================] - 0s 49ms/step - loss: 0.0102 - accuracy: 0.7013
#Epoch 26/40
#3/3 [==============================] - 0s 51ms/step - loss: 0.0174 - accuracy: 0.6940
#Epoch 27/40
#3/3 [==============================] - 0s 56ms/step - loss: 0.0127 - accuracy: 0.7088
#Epoch 28/40
#3/3 [==============================] - 0s 52ms/step - loss: 0.0115 - accuracy: 0.6919
#Epoch 29/40
#3/3 [==============================] - 0s 51ms/step - loss: 0.0107 - accuracy: 0.6947
#Epoch 30/40
#3/3 [==============================] - 0s 51ms/step - loss: 0.0099 - accuracy: 0.6949
#Epoch 31/40
#3/3 [==============================] - 0s 52ms/step - loss: 0.0098 - accuracy: 0.6729
#Epoch 32/40
#3/3 [==============================] - 0s 52ms/step - loss: 0.0088 - accuracy: 0.6999
#Epoch 33/40
#3/3 [==============================] - 0s 46ms/step - loss: 0.0085 - accuracy: 0.7314
#Epoch 34/40
#3/3 [==============================] - 0s 51ms/step - loss: 0.0085 - accuracy: 0.7139
#Epoch 35/40
#3/3 [==============================] - 0s 54ms/step - loss: 0.0082 - accuracy: 0.7314
#Epoch 36/40
#3/3 [==============================] - 0s 51ms/step - loss: 0.0081 - accuracy: 0.7278
#Epoch 37/40
#3/3 [==============================] - 0s 47ms/step - loss: 0.0080 - accuracy: 0.7415
#Epoch 38/40
#3/3 [==============================] - 0s 49ms/step - loss: 0.0076 - accuracy: 0.7496
#Epoch 39/40
#3/3 [==============================] - 0s 44ms/step - loss: 0.0079 - accuracy: 0.7446
#Epoch 40/40
#3/3 [==============================] - 0s 47ms/step - loss: 0.0075 - accuracy: 0.7423

# Model evaluation
result = model.evaluate(test_dataset, steps=10)

#10/10 [==============================] - 0s 22ms/step - loss: 0.0073 - accuracy: 0.7656
