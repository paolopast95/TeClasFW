import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from src.preprocessing.stemming import Stemmer
from src.preprocessing.tokenization import Tokenizer
from src.preprocessing.stopword_removal import StopwordRemoval
from tensorflow.keras.layers import Bidirectional, Dense, LSTM, Embedding
from tensorflow.keras import Model

class CustomizedLSTM(Model):
    def __init__(self, num_classes=1, num_hidden_layers=2, num_recurrent_units=[256,512], num_dense_layers=2, num_dense_neurons=[64], is_bidirectional=False, vocab_size=5000, pretrained_embeddings=None):
        super(CustomizedLSTM, self).__init__()
        self.lstms = []
        self.denses = []
        if not pretrained_embeddings is None:
            self.embedding = tf.keras.layers.Embedding(vocab_size, pretrained_embeddings.shape[1],
                                                       embeddings_initializer=tf.keras.initializers.Constant(pretrained_embeddings),
                                                       trainable=False)
        else:
            self.embedding = Embedding(vocab_size,300)
        if is_bidirectional:
            for i in range(num_hidden_layers - 1):
                self.lstms.append(Bidirectional(LSTM(num_recurrent_units[i], activation="relu", return_sequences=True)))
            self.lstms.append(Bidirectional(LSTM(num_recurrent_units[num_hidden_layers - 1], activation="relu", return_sequences=False)))
        else:
            for i in range(num_hidden_layers-1):
                self.lstms.append(LSTM(num_recurrent_units[i], activation="relu", return_sequences=True))
            self.lstms.append(LSTM(num_recurrent_units[num_hidden_layers-1], activation="relu", return_sequences=False))
        for i in range(num_dense_layers):
            self.denses.append(tf.keras.layers.Dense(num_dense_neurons[i], activation="relu"))
        self.classification_layer = tf.keras.layers.Dense(num_classes, activation="softmax")

    def call(self, inputs):
        x = self.embedding(inputs)
        for layer in self.lstms:
            x = layer(x)
        for layer in self.denses:
            x = layer(x)
        return self.classification_layer(x)
