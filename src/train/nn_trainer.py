import itertools
import os
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
import pandas as pd

from src.preprocessing.stemming import Stemmer
from src.preprocessing.stopword_removal import StopwordRemoval
from src.preprocessing.tokenization import Tokenizer
from src.models.cnn import CustomizedCNN

class NNTrainer:
    def __init__(self, model_name, params_dict, metric):
        self.model_name = model_name
        self.params_dict = params_dict
        self.metric = metric

    def compute_best_params(self, X_train, y_train, validation_size):
        self.best_accuracy = 0
        vocab_size = self.params_dict['vocab_size']
        X_concat = [" ".join(sentence) for sentence in X_train]
        tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="OOV")
        tokenizer.fit_on_texts(X)
        vocab_size = len(tokenizer.word_index) + 1
        sequences = tokenizer.texts_to_sequences(X)
        padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=30, padding='post', truncating='post')
        if "pretrained_embeddings_path" in self.params_dict:
            with open(os.path.join("../../embeddings", self.params_dict['pretrained_embeddings_path']),"rb") as f:
                pre_embedding_matrix = pickle.load(f)
            embedding_matrix = np.zeros((vocab_size, len(pre_embedding_matrix['cat'])))
            for word, i in tokenizer.word_index.items():
                if word in pre_embedding_matrix:
                    embedding_vector = pre_embedding_matrix[word]
                    embedding_matrix[i] = embedding_vector
            print(embedding_matrix.shape)
        else:
            embedding_matrix = None
        if self.model_name == "cnn":
            num_conv_layers = self.params_dict['num_conv_layers']
            num_conv_cells = self.params_dict['num_conv_cells']
            dim_filters = self.params_dict['dim_filters']
            pooling = (self.params_dict['pooling'])
            num_dense_layers = self.params_dict['num_dense_layers']
            num_dense_neurons = self.params_dict['num_dense_neurons']
            for ncl, ncc, df, p, ndl, ndn in itertools.product(num_conv_layers, num_conv_cells, dim_filters,pooling,num_dense_layers, num_dense_neurons):
                cnn = CustomizedCNN(num_classes=1, num_conv_layers=ncl, num_conv_cells=ncc, dim_filter=df, pooling=p, num_dense_layers=ndl, num_dense_neurons=ndn, pretrained_embeddings=embedding_matrix, vocab_size=vocab_size)
                cnn.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
                cnn.fit(padded, y_train, validation_split=validation_size, epochs=10)

data = pd.read_csv("../../data/clickbait_data.csv", sep="\t", header=None)
X = data[0]
y = data[1]
tokenizer = Tokenizer("wordpunct", True)
X = tokenizer.fit(X)
stopword = StopwordRemoval("english")
X = stopword.fit(X)
stemmer = Stemmer("english", "wordnet")
X = stemmer.fit(X)
trainer = NNTrainer("cnn", {'num_conv_layers':[2], 'num_conv_cells':[[128,128]], 'dim_filters':[[5,5,5]],'pooling':[[5,10]],
                            'num_dense_layers':[2], 'num_dense_neurons':[[32,16]], 'vocab_size':10000,
                            "pretrained_embeddings_path":"word2vec-google-news-300"}, "accuracy" )
trainer.compute_best_params(X,y,validation_size=0.2)

