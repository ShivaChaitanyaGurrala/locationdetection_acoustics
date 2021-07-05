import tensorflow as tf
import numpy as np


class ResnetIdentityBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters):
        super(ResnetIdentityBlock, self).__init__(name='')

        self.conv2a = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, padding="same")
        self.bn2a = tf.keras.layers.BatchNormalization()

        self.conv2b = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, padding='same')
        self.bn2b = tf.keras.layers.BatchNormalization()

        self.conv2c = tf.keras.layers.Conv2D(int(filters / 2), kernel_size=kernel_size, padding='same')
        self.bn2c = tf.keras.layers.BatchNormalization()

    def call(self, input_tensor):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=True)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=True)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=True)
        # concatinate the tensors over channels index
        x = tf.concat([x, input_tensor], axis=-1)
        return tf.nn.relu(x)


class DenseANN(tf.keras.Model):
    def __init__(self, no_neurons):
        super().__init__()
        self.dense_1 = tf.keras.layers.Dense(no_neurons, activation="relu")
        self.dense_2 = tf.keras.layers.Dense(no_neurons / 2, activation="relu")
        self.dense_3 = tf.keras.layers.Dense(no_neurons / 4, activation="relu")
        self.dense_4 = tf.keras.layers.Dense(no_neurons / 4, activation="relu")
        self.dense_5 = tf.keras.layers.Dense(no_neurons / 4, activation="relu")
        self.dense_6 = tf.keras.layers.Dense(no_neurons / 4, activation="relu")
        self.dense_final = tf.keras.layers.Dense(10, activation="softmax")
        self.drop = tf.keras.layers.Dropout(0.2)

    def call(self, inputs, training=None):
        x = tf.keras.layers.Flatten()(inputs)
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.dense_4(x)
        if training:
            x = self.drop(x)
        x = self.dense_5(x)
        if training:
            x = self.drop(x)
        x = self.dense_6(x)
        return self.dense_final(x)


class RCNN(tf.keras.Model):
    def __init__(self, kernel_size, filters):
        super().__init__()
        self.kernel_size = kernel_size
        self.filters = filters
        self.rrdb_1 = ResnetIdentityBlock(kernel_size=kernel_size, filters=self.filters)
        self.rrdb_2 = ResnetIdentityBlock(kernel_size=kernel_size, filters=self.filters)
        # self.rrdb_3 = ResnetIdentityBlock(kernel_size=kernel_size, filters=self.filters)

        self.conv_1 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=kernel_size, padding='same')
        self.conv_2 = tf.keras.layers.Conv2D(filters=self.filters / 4, kernel_size=kernel_size, padding='same')
        self.conv_3 = tf.keras.layers.Conv2D(filters=self.filters / 8, kernel_size=kernel_size, padding='same')
        self.flatten = tf.keras.layers.Flatten()
        self.relu = tf.keras.layers.LeakyReLU()
        self.dense_int = tf.keras.layers.Dense(512, activation='relu')
        self.dense_last = tf.keras.layers.Dense(10, activation='softmax')
        self.max_pool = tf.keras.layers.MaxPool2D()
        self.drop = tf.keras.layers.Dropout(0.2)

    def call(self, input_tensor):
        x = self.rrdb_1(input_tensor)
        x = self.rrdb_2(x)
        # x = self.rrdb_3(x)
        x = self.relu(self.max_pool(x))
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.conv_2(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.conv_3(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.flatten(x)
        x = self.dense_int(x)
        # Ten neurons as we get the probabilities of all the ten classes
        out = self.dense_last(x)

        return out


'''
rcnn = RCNN(filters=32, kernel_size=3)
arr = tf.zeros([4, 256, 256, 3], dtype=tf.float32)
rcnn.build(arr.shape)
rcnn.summary()
'''
