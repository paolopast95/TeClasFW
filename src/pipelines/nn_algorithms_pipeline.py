from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.utils import np_utils

from src.preprocessing.sentence_embedding import Vectorizer
from src.preprocessing.stemming import Stemmer
from src.preprocessing.stopword_removal import StopwordRemoval
from src.preprocessing.tokenization import Tokenizer
from src.train.classical_trainer import ClassicalTrainer

import pandas as pd
import yaml
import os
import warnings

from src.train.nn_trainer import NNTrainer


def run_dl_algorithms(config_filename):
    with open(os.path.join("../../config_files", config_filename), "r") as f:
        config = yaml.safe_load(f)['experiment']
    print(config)
    dataset_filename = config['dataset_filename']
    output_folder_name = config['output_folder_name']
    tokenization = config['preprocessing']["tokenization"]
    stemming = config['preprocessing']["stemming"]
    stopword_removal = config['preprocessing']["stopword_removal"]
    validation_size = config['evaluation']['validation_set']['validation_size']
    test_size = config['evaluation']['validation_set']['test_size']
    max_sentence_length = config['max_sentence_length']
    metrics = config['evaluation']['metrics']
    models = config['models']
    dataset = pd.read_csv(os.path.join("../../data", dataset_filename), header=None, sep="\t")
    X = dataset[0]
    y = dataset[1]
    tokenizer = Tokenizer(tokenization, True)
    X = tokenizer.fit(X)
    if stopword_removal:
        stopword = StopwordRemoval("english")
        X = stopword.fit(X)
    stemmer = Stemmer("english", stemming)
    X = stemmer.fit(X)
    print(y.value_counts())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)
    encoder = LabelEncoder()
    encoder.fit(y)
    y_train = encoder.transform(y_train)
    y_test = encoder.transform(y_test)
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    for model in models:
        trainer = NNTrainer(metrics=metrics, output_folder_name=output_folder_name, model_name=model['model_name'], params_dict=model['params'])
        trainer.compute_best_params(X_train, y_train, X_test, y_test, validation_size=validation_size, max_sentence_length=max_sentence_length)


run_dl_algorithms("nn_algorithms.yml")