import tensorflow as tf

class TestLSTM(tf.keras.Model):
    def __init__(self, units, name=None):
        super(TestLSTM, self).__init__(name=name)
        self.lstm_layer = tf.keras.layers.LSTM(units, activation='softsign')
        self.dense = tf.keras.layers.Dense(1)
        self.sigmoid = tf.keras.layers.Activation('sigmoid')

    def call(self, x):
        sequence = self.lstm_layer(x)
        logit = self.dense(sequence)
        prob = self.sigmoid(logit)

        return prob

    def get_sequence(self, x):
        return self.lstm_layer(x)