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
    if config['preprocessing']['embedding_strategy'] == "word2vec":
        embedding_file = config['preprocessing']["embedding_file"]
    else:
        embedding_file = None
    models = config['models']
    dataset = pd.read_csv(os.path.join("../../data", dataset_filename), header=None, sep="\t")
    X = dataset[0]
    y = dataset[1]
    print(type(y[0]))
    tokenizer = Tokenizer(tokenization, True)
    X = tokenizer.fit(X)
    if stopword_removal:
        stopword = StopwordRemoval("english")
        X = stopword.fit(X)
    stemmer = Stemmer("english", stemming)
    X = stemmer.fit(X)

    print(models)
    for model in models:
        trainer = NNTrainer(metric="accuracy", output_folder_name=output_folder_name, model_name=model['model_name'], params_dict=model['params'])
        trainer.compute_best_params(X, y, validation_size=0.2)


run_dl_algorithms("nn_algorithms.yml")