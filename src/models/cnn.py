import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from src.preprocessing.stemming import Stemmer
from src.preprocessing.tokenization import Tokenizer
from src.preprocessing.stopword_removal import StopwordRemoval
from tensorflow.keras.layers import Dense, Flatten, MaxPooling1D, Conv1D, Embedding
from tensorflow.keras import Model

class CustomizedCNN(Model):
    def __init__(self, num_classes=1, num_conv_layers=2, num_conv_cells=[256,512], dim_filter=[5,5], pooling=[5,25], num_dense_layers=2, num_dense_neurons=[64],
                 vocab_size=5000, pretrained_embeddings=None):
        super(CustomizedCNN, self).__init__()
        self.cnns = []
        self.maxpoolings = []
        self.denses = []
        if pretrained_embeddings:
            self.embedding = tf.keras.layers.Embedding(vocab_size, pretrained_embeddings.shape(0),
                                                       embeddings_initializer=tf.keras.initializers.Constant(pretrained_embeddings),
                                                       trainable=False)
        else:
            self.embedding = Embedding(vocab_size,300)
        for i in range(num_conv_layers):
            self.cnns.append(Conv1D(num_conv_cells[i], dim_filter[i],activation="relu", padding="same"))
            self.maxpoolings.append(MaxPooling1D(pooling[i], padding="same"))
        self.flatten = Flatten()
        for i in range(num_dense_layers):
            self.denses.append(tf.keras.layers.Dense(num_dense_neurons[i], activation="relu"))
        self.classification_layer = tf.keras.layers.Dense(num_classes, activation="sigmoid")

    def call(self, inputs):
        x = self.embedding(inputs)
        print(x.shape)
        for cnn, pooling in zip(self.cnns, self.maxpoolings):
            x = cnn(x)
            x = pooling(x)
        x = self.flatten(x)
        for layer in self.denses:
            x = layer(x)
        return self.classification_layer(x)


data = pd.read_csv("../../data/clickbait_data.csv", sep="\t", header=None)
X = data[0]
y = data[1]
print(type(y[0]))
tokenizer = Tokenizer("wordpunct", True)
X = tokenizer.fit(X)
stopword = StopwordRemoval("english")
X = stopword.fit(X)
stemmer = Stemmer("english", "wordnet")
X = stemmer.fit(X)
X = [" ".join(sent) for sent in X]
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words = 5000, oov_token="OOV")
tokenizer.fit_on_texts(X)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(X)
padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=30,padding='post', truncating='post')
encoder = LabelEncoder()
encoder.fit(y)
print(y[0])
print(padded[0])
print(padded.shape)
lstm = CustomizedCNN(num_classes=1, num_conv_layers=3, num_conv_cells=[128,128,128], dim_filter=[5,5,5], pooling=[5,5,5], num_dense_layers=3, num_dense_neurons=[128,64,32], vocab_size=5000)
lstm.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
lstm.fit(padded,y,validation_split=0.2, epochs=10)
print(lstm.layers)