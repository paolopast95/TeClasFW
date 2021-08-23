import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
import itertools
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
import pandas as pd
from src.preprocessing.stemming import Stemmer
from src.preprocessing.tokenization import Tokenizer
from src.preprocessing.stopword_removal import StopwordRemoval
from src.preprocessing.sentence_embedding import Vectorizer
from tqdm import tqdm

class CustomizedLSTM(tf.keras.Model):
    def __init__(self, num_classes=2, num_hidden_layers=2, num_recurrent_units=[256,512], num_dense_layers=2, num_dense_neurons=[64], is_bidirectional=False, vocab_size=5000):
        super(CustomizedLSTM, self).__init__()
        self.lstms = []
        self.embedding = tf.keras.layers.Embedding(vocab_size,300)
        for i in range(num_hidden_layers):
            self.lstms.append(tf.keras.layers.LSTM(num_recurrent_units[i], activation="relu", return_sequences=True))
        self.lstms.append(tf.keras.layers.LSTM(num_recurrent_units[0], activation="relu", return_sequences=False))
        self.first_dense = tf.keras.layers.Dense(num_dense_neurons[0], activation="relu")
        self.classification_layer = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs):
        x = self.embedding(inputs)
        for layer in self.lstms:
            x = layer(x)
        x = self.first_dense(x)
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
lstm = CustomizedLSTM(vocab_size=5000)
lstm.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
lstm.fit(padded,y,validation_split=0.2, epochs=10)
print(lstm.layers)