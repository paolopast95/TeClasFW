import itertools
import os
import pickle

import numpy as np
import tensorflow as tf
import gensim.downloader as gensim_api
from src.models.gru import CustomizedGRU
from src.models.lstm import CustomizedLSTM
from src.models.rnn import CustomizedRNN

from src.models.cnn import CustomizedCNN

class NNTrainer:
    def __init__(self, model_name, params_dict, metric):
        self.model_name = model_name
        self.params_dict = params_dict
        self.metric = metric

    def compute_best_params(self, X_train, y_train, validation_size):
        self.best_accuracy = 0
        X_concat = [" ".join(sentence) for sentence in X_train]
        tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="OOV")
        tokenizer.fit_on_texts(X_concat)
        vocab_size = len(tokenizer.word_index) + 1
        sequences = tokenizer.texts_to_sequences(X_concat)
        padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=30, padding='post', truncating='post')
        if "pretrained_embeddings_path" in self.params_dict:
            if os.path.exists(os.path.join("../../embeddings/", self.params_dict['pretrained_embeddings_path'])):
                with open(os.path.join("../../embeddings", self.params_dict['pretrained_embeddings_path']),"rb") as f:
                    pre_embedding_matrix = pickle.load(f)
            else:
                nlp = gensim_api.load("word2vec-google-news-300")
                with open(os.path.join("../../embeddings/", self.params_dict['pretrained_embeddings_path']), "wb") as f:
                    pickle.dump(nlp, f)
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
        else:
            num_hidden_layers = self.params_dict['num_hidden_layers']
            num_recurrent_units = self.params_dict['num_recurrent_units']
            num_dense_layers = self.params_dict['num_dense_layers']
            num_dense_neurons = self.params_dict['num_dense_neurons']
            is_bidirectional = self.params_dict['is_bidirectional']
            for nhl, nru, ndl, ndn, bi in itertools.product(num_hidden_layers, num_recurrent_units, num_dense_layers, num_dense_neurons, is_bidirectional):
                if self.model_name == "lstm":
                    model = CustomizedLSTM(num_classes=1, num_hidden_layers=nhl, num_recurrent_units=nru,
                                           num_dense_layers=ndl, num_dense_neurons=ndn, is_bidirectional=bi, pretrained_embeddings=embedding_matrix)
                elif self.model_name == "rnn":
                    model = CustomizedRNN(num_classes=1, num_hidden_layers=nhl, num_recurrent_units=nru,
                                           num_dense_layers=ndl, num_dense_neurons=ndn, is_bidirectional=bi, pretrained_embeddings=embedding_matrix)
                elif self.model_name == "gru":
                    model = CustomizedGRU(num_classes=1, num_hidden_layers=nhl, num_recurrent_units=nru,
                                           num_dense_layers=ndl, num_dense_neurons=ndn, is_bidirectional=bi, pretrained_embeddings=embedding_matrix)
                model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
                model.fit(padded, y_train, validation_split=validation_size, epochs=10)



