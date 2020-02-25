import tensorflow as tf
import numpy as np


class LanguageCNN(tf.keras.Model):
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 num_filters: int,
                 embedding_matrix: np.ndarray,
                 maxlen: int,
                 name: str = None):
        super(LanguageCNN, self).__init__(name=name)
        self.embeddings = tf.keras.layers.Embedding(vocab_size,
                                                    embedding_dim,
                                                    embeddings_initializer=tf.keras.initializers.Constant(
                                                        embedding_matrix),
                                                    input_length=maxlen,
                                                    trainable=True)
        self.reshape = tf.keras.layers.Reshape((maxlen, embedding_dim, 1))

        self.conv_0 = tf.keras.layers.Conv2D(num_filters,
                                             kernel_size=(3, embedding_dim),
                                             activation='relu',
                                             kernel_regularizer=tf.keras.regularizers.l2(
                                                 3))
        self.conv_1 = tf.keras.layers.Conv2D(num_filters,
                                             kernel_size=(4, embedding_dim),
                                             activation='relu',
                                             kernel_regularizer=tf.keras.regularizers.l2(
                                                 3))
        self.conv_2 = tf.keras.layers.Conv2D(num_filters,
                                             kernel_size=(5, embedding_dim),
                                             activation='relu',
                                             kernel_regularizer=tf.keras.regularizers.l2(
                                                 3))

        self.maxpool_0 = tf.keras.layers.MaxPool2D(
            pool_size=(maxlen - 3 + 1, 1),
            strides=(1, 1),
            padding='valid')
        self.maxpool_1 = tf.keras.layers.MaxPool2D(
            pool_size=(maxlen - 4 + 1, 1),
            strides=(1, 1),
            padding='valid')
        self.maxpool_2 = tf.keras.layers.MaxPool2D(
            pool_size=(maxlen - 5 + 1, 1),
            strides=(1, 1),
            padding='valid')

        self.concat = tf.keras.layers.Concatenate(axis=1)
        self.flatten = tf.keras.layers.Flatten()

        self.dropout = tf.keras.layers.Dropout(0.5)
        self.out = tf.keras.layers.Dense(units=1, activation='sigmoid')

    def call(self, x):
        embeddings = self.embeddings(x)
        return self.embedded(embeddings)

    def embedded(self, embeddings):
        reshape = self.reshape(embeddings)

        conv_0 = self.conv_0(reshape)
        conv_1 = self.conv_1(reshape)
        conv_2 = self.conv_2(reshape)

        maxpool_0 = self.maxpool_0(conv_0)
        maxpool_1 = self.maxpool_1(conv_1)
        maxpool_2 = self.maxpool_2(conv_2)

        concat = self.concat([maxpool_0, maxpool_1, maxpool_2])
        flatten = self.flatten(concat)

        dropout = self.dropout(flatten)
        out = self.out(dropout)
        return out
