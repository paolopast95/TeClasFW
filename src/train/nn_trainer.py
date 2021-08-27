import itertools
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import gensim.downloader as gensim_api
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.python.keras.utils import np_utils

from src.models.gru import CustomizedGRU
from src.models.lstm import CustomizedLSTM
from src.models.rnn import CustomizedRNN
from src.models.cnn import CustomizedCNN

from tensorflow.keras.optimizers import Adadelta, Adagrad, Adam, Adamax, Ftrl, Nadam, RMSprop, SGD

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

metrics_dict = {
    "accuracy": accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1": f1_score
}
class NNTrainer:
    def __init__(self, output_folder_name, model_name, params_dict, metrics):
        self.output_folder_name = output_folder_name
        self.model_name = model_name
        self.params_dict = params_dict
        self.metrics = metrics

    def compute_best_params(self, X_train, y_train, X_test, y_test, validation_size, max_sentence_length):
        output_folder = os.path.join("../../output/", self.output_folder_name)
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        best_accuracy = 0
        learning_rates = self.params_dict['learning_rates']
        optimizers = self.params_dict['optimizers']
        epochs = self.params_dict['epochs']
        loss = self.params_dict['loss']
        X_concat_train = [" ".join(sentence) for sentence in X_train]
        X_concat_test = [" ".join(sentence) for sentence in X_test]
        tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="OOV")
        tokenizer.fit_on_texts(X_concat_train)
        vocab_size = len(tokenizer.word_index) + 1
        sequences_train = tokenizer.texts_to_sequences(X_concat_train)
        sequences_test = tokenizer.texts_to_sequences(X_concat_test)
        padded_train = tf.keras.preprocessing.sequence.pad_sequences(sequences_train, maxlen=max_sentence_length, padding='post', truncating='post')
        padded_test = tf.keras.preprocessing.sequence.pad_sequences(sequences_test, maxlen=max_sentence_length, padding='post',
                                                               truncating='post')
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
            results = pd.DataFrame(columns=["NumConvLayers", "NumFilters", "KernelDimensions", "NumDenseLayers", "NumDenseNeurons", "Optimizer", "LearningRate", "Epochs", "Loss", "ValidationLoss"] + self.metrics)
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

                cnn = CustomizedCNN(num_classes=y_train.shape[1], num_conv_layers=ncl, num_conv_cells=ncc, dim_filter=df, pooling=p, num_dense_layers=ndl, num_dense_neurons=ndn, pretrained_embeddings=embedding_matrix, vocab_size=vocab_size)
                cnn.compile(loss=loss, optimizer=optimizers_dict[opt](learning_rate=lr), metrics=["accuracy"])
                early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, verbose=0, mode='max')
                mcp_save = ModelCheckpoint('../../temp_models/.mdl_wts.hdf5', save_best_only=True, monitor='val_accuracy', mode='max')
                history = cnn.fit(padded_train, y_train, validation_split=validation_size, epochs=ep, callbacks=[early_stopping, mcp_save])
                cnn.load_weights("../../temp_models/.mdl_wts.hdf5")
                y_pred = [np.argmax(el) for el in cnn.predict(padded_test)]
                current_experiment_results = [ncl, ncc, df, ndl, ndn, opt, lr, ep, history.history['loss'][-1], history.history['val_loss'][-1]]
                for metric in self.metrics:
                    if metric == "precision":
                        current_metric_value = precision_score(np.argmax(y_test, axis=1),y_pred, average="macro")
                        print("Test Precision Score: " + str(current_metric_value))
                    elif metric == "recall":
                        current_metric_value = recall_score(np.argmax(y_test, axis=1), y_pred, average="macro")
                        print("Test Recall Score: " + str(current_metric_value))
                    elif metric == "f1":
                        current_metric_value = f1_score(np.argmax(y_test, axis=1), y_pred, average="macro")
                        print("Test F1 Score: " + str(current_metric_value))
                    else:
                        current_metric_value = accuracy_score(np.argmax(y_test, axis=1), y_pred)
                        print("Test Accuracy Score: " + str(current_metric_value))
                    current_experiment_results.append(current_metric_value)
                print("-------------------------------------------------------------------------")

                results.loc[experiment_number] = current_experiment_results
                experiment_number += 1
            results.to_csv(os.path.join(output_folder, "cnn.csv"))

        else:
            num_hidden_layers = self.params_dict['num_hidden_layers']
            num_recurrent_units = self.params_dict['num_recurrent_units']
            num_dense_layers = self.params_dict['num_dense_layers']
            num_dense_neurons = self.params_dict['num_dense_neurons']
            is_bidirectional = self.params_dict['is_bidirectional']
            experiment_number = 0
            results = pd.DataFrame(columns=["NumRecLayers", "NumRecUnits", "NumDenseLayers", "NumDenseNeurons", "IsBidirectional", "Optimizer", "LearningRate", "Epochs", "Loss", "ValidationLoss"]+ self.metrics)
            for nhl, nru, ndl, ndn, bi, opt, lr, ep in itertools.product(num_hidden_layers, num_recurrent_units, num_dense_layers, num_dense_neurons, is_bidirectional, optimizers, learning_rates, epochs):
                tf.keras.backend.clear_session()
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
                print(y_train.shape)
                if self.model_name == "lstm":
                    model = CustomizedLSTM(num_classes=y_train.shape[1], num_hidden_layers=nhl, num_recurrent_units=nru,
                                           num_dense_layers=ndl, num_dense_neurons=ndn, is_bidirectional=bi, pretrained_embeddings=embedding_matrix, vocab_size=vocab_size)
                elif self.model_name == "rnn":
                    model = CustomizedRNN(num_classes=y_train.shape[1], num_hidden_layers=nhl, num_recurrent_units=nru,
                                           num_dense_layers=ndl, num_dense_neurons=ndn, is_bidirectional=bi, pretrained_embeddings=embedding_matrix, vocab_size=vocab_size)
                elif self.model_name == "gru":
                    model = CustomizedGRU(num_classes=y_train.shape[1], num_hidden_layers=nhl, num_recurrent_units=nru,
                                           num_dense_layers=ndl, num_dense_neurons=ndn, is_bidirectional=bi, pretrained_embeddings=embedding_matrix,  vocab_size=vocab_size)
                model.compile(loss=loss, optimizer=optimizers_dict[opt](learning_rate=lr), metrics=['accuracy'])
                early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, verbose=0, mode='max')
                mcp_save = ModelCheckpoint('../../temp_models/.mdl_wts.hdf5', save_best_only=True,
                                           monitor='val_accuracy', mode='max')
                history = model.fit(padded_train, y_train, validation_split=validation_size, epochs=ep,
                                  callbacks=[early_stopping, mcp_save])
                model.load_weights("../../temp_models/.mdl_wts.hdf5")
                y_pred = [np.argmax(el) for el in model.predict(padded_test)]
                current_experiment_results = [nhl, nru, ndl, ndn, bi, opt, lr, ep, history.history['loss'][-1],
                                              history.history['val_loss'][-1]]
                for metric in self.metrics:
                    if metric == "precision":
                        current_metric_value = precision_score(np.argmax(y_test, axis=1), y_pred, average="macro")
                        print("Test Precision Score: " + str(current_metric_value))
                    elif metric == "recall":
                        current_metric_value = recall_score(np.argmax(y_test, axis=1), y_pred, average="macro")
                        print("Test Recall Score: " + str(current_metric_value))
                    elif metric == "f1":
                        current_metric_value = f1_score(np.argmax(y_test, axis=1), y_pred, average="macro")
                        print("Test F1 Score: " + str(current_metric_value))
                    else:
                        current_metric_value = accuracy_score(np.argmax(y_test, axis=1), y_pred)
                        print("Test Accuracy Score: " + str(current_metric_value))
                    current_experiment_results.append(current_metric_value)
                print("-------------------------------------------------------------------------")
                experiment_number += 1
            results.to_csv(os.path.join(output_folder, self.model_name+".csv"))



