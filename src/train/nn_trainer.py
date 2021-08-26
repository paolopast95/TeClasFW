import itertools
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import gensim.downloader as gensim_api
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.python.keras.utils import np_utils

from src.models.gru import CustomizedGRU
from src.models.lstm import CustomizedLSTM
from src.models.rnn import CustomizedRNN
from src.models.cnn import CustomizedCNN

from tensorflow.keras.optimizers import Adadelta, Adagrad, Adam, Adamax, Ftrl, Nadam, RMSprop, SGD
from tensorflow.keras.models import load_model

tf.random.set_seed(42)

optimizers_dict = {
    "adadelta": Adadelta,
    "adagrad": Adagrad,
    "adam": Adam,
    "adamax": Adamax,
    "ftrl": Ftrl,
    "nadam": Nadam,
    "rmsprop": RMSprop,
    "sgd": SGD
}
class NNTrainer:
    def __init__(self, output_folder_name, model_name, params_dict, metric):
        self.output_folder_name = output_folder_name
        self.model_name = model_name
        self.params_dict = params_dict
        self.metric = metric

    def compute_best_params(self, X_train, y_train, validation_size):
        encoder = LabelEncoder()
        encoder.fit(y_train)
        encoded_Y = encoder.transform(y_train)
        dummy_y = np_utils.to_categorical(encoded_Y)
        print(dummy_y)
        output_folder = os.path.join("../../output/", self.output_folder_name)
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        best_accuracy = 0
        learning_rates = self.params_dict['learning_rates']
        optimizers = self.params_dict['optimizers']
        epochs = self.params_dict['epochs']
        loss = self.params_dict['loss']
        print(X_train[0])
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
        else:
            embedding_matrix = None
        if self.model_name == "cnn":
            experiment_number = 0
            num_conv_layers = self.params_dict['num_conv_layers']
            num_conv_cells = self.params_dict['num_conv_cells']
            dim_filters = self.params_dict['dim_filters']
            pooling = (self.params_dict['pooling'])
            num_dense_layers = self.params_dict['num_dense_layers']
            num_dense_neurons = self.params_dict['num_dense_neurons']
            results = pd.DataFrame(columns=["NumConvLayers", "NumFilters", "KernelDimensions", "NumDenseLayers", "NumDenseNeurons", "Optimizer", "LearningRate", "Epochs", "Loss", "ValidationLoss", self.metric.title(), "Validation"+self.metric.title()])
            for ncl, ncc, df, p, ndl, ndn, opt, lr, ep in itertools.product(num_conv_layers, num_conv_cells, dim_filters,pooling,num_dense_layers, num_dense_neurons, optimizers, learning_rates, epochs):
                print("Training and validation of "+self.model_name.upper()+" with the following parameters with early stopping")
                print("Number of Convolutional Layers: " + str(ncl))
                print("Number of Filters per Layer: " + str(ncc))
                print("Kernel Size per Layer: " + str(df))
                print("Number of Dense Layers: " + str(ndl))
                print("Number of Dense Neurons per Layer: " + str(ndn))
                print("Optimizer: " + str(opt))
                print("Learning Rate: " + str(lr))
                print("Epochs: " + str(ep))
                print("Loss Function: " + str(loss))

                cnn = CustomizedCNN(num_classes=dummy_y.shape[1], num_conv_layers=ncl, num_conv_cells=ncc, dim_filter=df, pooling=p, num_dense_layers=ndl, num_dense_neurons=ndn, pretrained_embeddings=embedding_matrix, vocab_size=vocab_size)
                cnn.compile(loss=loss, optimizer=optimizers_dict[opt](learning_rate=lr), metrics=[self.metric])
                early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, verbose=0, mode='max')
                mcp_save = ModelCheckpoint('../../temp_models/.mdl_wts.hdf5', save_best_only=True, monitor='val_accuracy', mode='max')
                history = cnn.fit(padded, dummy_y, validation_split=validation_size, epochs=ep, callbacks=[early_stopping, mcp_save])
                cnn.load_weights("../../temp_models/.mdl_wts.hdf5")
                y_pred = [np.argmax(el) for el in cnn.predict(padded)]
                current_accuracy = accuracy_score(encoded_Y, y_pred)
                print("Test Accuracy: " + str(current_accuracy))
                print("-------------------------------------------------------------------------")

                results.loc[experiment_number] = [ncl, ncc, df, ndl, ndn, opt, lr, ep, history.history['loss'][-1], history.history['val_loss'][-1], history.history['accuracy'][-1], history.history["val_accuracy"][-1]]
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    best_model = cnn
                experiment_number += 1
            results.to_csv(os.path.join(output_folder, "cnn.csv"))

        else:
            num_hidden_layers = self.params_dict['num_hidden_layers']
            num_recurrent_units = self.params_dict['num_recurrent_units']
            num_dense_layers = self.params_dict['num_dense_layers']
            num_dense_neurons = self.params_dict['num_dense_neurons']
            is_bidirectional = self.params_dict['is_bidirectional']
            experiment_number = 0
            results = pd.DataFrame(columns=["NumRecLayers", "NumRecUnits", "NumDenseLayers", "NumDenseNeurons", "IsBidirectional", "Optimizer", "LearningRate", "Epochs", "Loss", "ValidationLoss", self.metric.title(), "Validation"+self.metric.title()])
            for nhl, nru, ndl, ndn, bi, opt, lr, ep in itertools.product(num_hidden_layers, num_recurrent_units, num_dense_layers, num_dense_neurons, is_bidirectional, optimizers, learning_rates, epochs):
                print("Training and validation of " + self.model_name.upper() + " with the following parameters")
                print("Number of Recurrent Layers: " + str(nhl))
                print("Number of Recurrent Units per Layer: " + str(nru))
                print("Number of Dense Layers: " + str(ndl))
                print("Number of Dense Neurons per layer: " + str(ndn))
                print("Is Bidirectional: " + str(bi))
                print("Optimizer: " + str(opt))
                print("Learning Rate: " + str(lr))
                print("Epochs: " + str(ep))
                print("Loss Function: " + str(loss))
                if self.model_name == "lstm":
                    model = CustomizedLSTM(num_classes=1, num_hidden_layers=nhl, num_recurrent_units=nru,
                                           num_dense_layers=ndl, num_dense_neurons=ndn, is_bidirectional=bi, pretrained_embeddings=embedding_matrix)
                elif self.model_name == "rnn":
                    model = CustomizedRNN(num_classes=1, num_hidden_layers=nhl, num_recurrent_units=nru,
                                           num_dense_layers=ndl, num_dense_neurons=ndn, is_bidirectional=bi, pretrained_embeddings=embedding_matrix)
                elif self.model_name == "gru":
                    model = CustomizedGRU(num_classes=1, num_hidden_layers=nhl, num_recurrent_units=nru,
                                           num_dense_layers=ndl, num_dense_neurons=ndn, is_bidirectional=bi, pretrained_embeddings=embedding_matrix)
                model.compile(loss=loss, optimizer=optimizers_dict[opt](learning_rate=lr), metrics=[self.metric])
                history = model.fit(padded, y_train, validation_split=validation_size, epochs=10, verbose=0)
                early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, verbose=0, mode='max')
                mcp_save = ModelCheckpoint('../../temp_models/.mdl_wts.hdf5', save_best_only=True,
                                           monitor='val_accuracy', mode='max')
                history = model.fit(padded, dummy_y, validation_split=validation_size, epochs=ep,
                                  callbacks=[early_stopping, mcp_save])
                model.load_weights("../../temp_models/.mdl_wts.hdf5")
                y_pred = [np.argmax(el) for el in model.predict(padded)]
                current_accuracy = accuracy_score(encoded_Y, y_pred)
                print("Test Accuracy: " + str(current_accuracy))
                print("-------------------------------------------------------------------------")
                results.loc[experiment_number] = [nhl, nru, ndl, ndn, bi, opt, lr, ep,
                                                   history.history['loss'][epochs - 1],
                                                   history.history['val_loss'][epochs - 1],
                                                   history.history[self.metric][epochs - 1],
                                                   history.history["val_" + self.metric][epochs - 1]]
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    best_model = model
                experiment_number += 1
                experiment_number += 1
            results.to_csv(os.path.join(output_folder, self.model_name+".csv"))



