import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, Dense, RNN, Embedding
from tensorflow.keras import Model

class CustomizedRNN(Model):
    def __init__(self, num_classes=1, num_hidden_layers=2, num_recurrent_units=[256,512], num_dense_layers=2, num_dense_neurons=[64], is_bidirectional=False, vocab_size=5000, pretrained_embeddings=None):
        super(CustomizedRNN, self).__init__()
        self.rnns = []
        self.denses = []
        if pretrained_embeddings:
            self.embedding = tf.keras.layers.Embedding(vocab_size, pretrained_embeddings.shape(0),
                                                       embeddings_initializer=tf.keras.initializers.Constant(pretrained_embeddings),
                                                       trainable=False)
        else:
            self.embedding = Embedding(vocab_size,300)
        if is_bidirectional:
            for i in range(num_hidden_layers - 1):
                self.rnns.append(Bidirectional(RNN(num_recurrent_units[i], activation="relu", return_sequences=True)))
            self.rnns.append(Bidirectional(RNN(num_recurrent_units[num_hidden_layers - 1], activation="relu", return_sequences=False)))
        else:
            for i in range(num_hidden_layers-1):
                self.rnns.append(RNN(num_recurrent_units[i], activation="relu", return_sequences=True))
            self.rnns.append(RNN(num_recurrent_units[num_hidden_layers-1], activation="relu", return_sequences=False))
        for i in range(num_dense_layers):
            self.denses.append(Dense(num_dense_neurons[i], activation="relu"))
        self.classification_layer = Dense(num_classes, activation="sigmoid")

    def call(self, inputs):
        x = self.embedding(inputs)
        for layer in self.rnns:
            x = layer(x)
        for layer in self.denses:
            x = layer(x)
        return self.classification_layer(x)