import tensorflow as tf
import tensorflow_datasets as tfds
mnist_data = tfds.load(name="fashion_mnist", split="train")

assert isinstance(mnist_data, tf.data.Dataset)
print(type(mnist_data))