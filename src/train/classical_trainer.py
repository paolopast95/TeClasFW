import os

from sklearn.feature_extraction.text import TfidfVectorizer
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

class ClassicalTrainer():
    def __init__(self, output_folder_name, model_name, params_dict, metric):
        self.output_folder_name = output_folder_name
        self.model_name = model_name
        self.params_dict = params_dict
        self.metric = metric


    def compute_best_params(self, X_train, y_train, validation_size):
        output_folder = os.path.join("../../output/", self.output_folder_name)
        self.best_accuracy = 0
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=validation_size)
        if self.model_name == "svm":
            kernels = self.params_dict['kernels']
            gammas = self.params_dict['gammas']
            Cs = self.params_dict['Cs']
            degrees = self.params_dict['degrees']
            results = pd.DataFrame(columns=["Kernel", "Gamma", "C", "Degree", "Accuracy"])
            experiment_number = 0
            for kernel, gamma, C, degree in tqdm(itertools.product(kernels, gammas, Cs, degrees)):
                print("Training and validation of SVM with the following parameters")
                print("Kernel: " + str(kernel))
                print("Gamma: " + str(gamma))
                print("C: " + str(C))
                print("Degree: " + str(degree))
                current_svm = SVC(kernel=kernel, gamma=gamma, C=C, degree=degree)
                current_svm.fit(X_tr, y_tr)
                y_pred = current_svm.predict(X_val)
                current_accuracy = accuracy_score(y_val, y_pred)
                if current_accuracy > self.best_accuracy:
                    self.best_model = current_svm
                    self.best_accuracy = current_accuracy
                print("Accuracy: " + str(current_accuracy))
                print("-------------------------------------------------------------------------")
                results.iloc[0] = [kernel, gamma, C, degree, current_accuracy]
                experiment_number += 1
            results.to_csv(os.path.join(output_folder, "svm.csv"))
        elif self.model_name == "naive_bayes":
            current_nb = MultinomialNB()
            current_nb.fit(X_tr,y_tr)
            y_pred = current_nb.predict(X_val)
            current_accuracy = accuracy_score(y_val, y_pred)
            self.best_model = current_nb
            self.best_accuracy = current_accuracy
        elif self.model_name == 'decision_tree':
            criteria = self.params_dict['criteria']
            splitters = self.params_dict['splitters']
            max_depths = self.params_dict['max_depths']
            min_samples_splits = self.params_dict['min_samples_splits']
            min_samples_leaves = self.params_dict['min_samples_leaves']
            results = pd.DataFrame(columns=["Criterion", "Splitter", "MaxDepth", "MinSampleSplit", "MinSampleLeaf", "Accuracy"])
            experiment_number = 0
            for criterion, splitter, max_depth, min_samples_split, min_samples_leaf in itertools.product(criteria, splitters, max_depths, min_samples_splits,min_samples_leaves):
                print("Training and validation of DecisionTree with the following parameters")
                print("Criterion: " + str(criterion))
                print("Splitter: " + str(splitter))
                print("Max Depth: " + str(max_depth))
                print("Min Sample Split: " + str(min_samples_split))
                print("Min Sample Leaf: ", str(min_samples_leaf))
                current_tree = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth, min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf)
                current_tree.fit(X_tr, y_tr)
                y_pred = current_tree.predict(X_val)
                current_accuracy = accuracy_score(y_val, y_pred)
                if current_accuracy > self.best_accuracy:
                    self.best_model = current_tree
                    self.best_accuracy = current_accuracy
                print("Accuracy: " + str(current_accuracy))
                print("-------------------------------------------------------------------------")
                results.iloc[experiment_number] = [criterion, splitter, max_depth, min_samples_split, min_samples_leaf, current_accuracy]
            results.to_csv(os.path.join(output_folder, "decisionTree.csv"))
        self.best_model.fit(X_train, y_train)
















    """
    def compute_best_params_cv(self, X_train, y_train, cv=5):
        self.best_accuracy = 0
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=420)
        if self.model_name == "svm":
            kernels = self.params_dict['kernels']
            gammas = self.params_dict['gammas']
            Cs = self.params_dict['Cs']
            degrees = self.params_dict['degrees']
            for kernel, gamma, C, degree in tqdm(itertools.product(kernels, gammas, Cs, degrees)):
                accuracy = 0
                current_svm = SVC(kernel=kernel, gamma=gamma, C=C, degree=degree)
                for train_index, test_index in skf.split(X_train,y_train):
                    X_tr, X_val = X_train[train_index], X_train[test_index]
                    y_tr, y_val = y_train[train_index], X_train[test_index]
                    current_svm.fit(X_train, y_train)
                    y_pred = current_svm.predict(X_val)
                    accuracy += accuracy_score(y_val, y_pred)
                accuracy /= cv
                if accuracy > self.best_accuracy:
                    self.best_model = current_svm
                    self.best_accuracy = accuracy
        elif self.model_name == "naive_bayes":
            accuracy = 0
            current_nb = MultinomialNB()
            for train_index, test_index in skf.split(X_train, y_train):
                X_tr, X_val = X_train[train_index], X_train[test_index]
                y_tr, y_val = y_train[train_index], X_train[test_index]
                current_nb.fit(X_tr,y_tr)
                y_pred = current_nb.predict(X_val)
                accuracy += accuracy_score(y_val, y_pred)
            accuracy /= cv
            self.best_model = current_nb
            self.best_accuracy = accuracy
        elif self.model_name == 'decision_tree':
            criteria = self.params_dict['criteria']
            splitters = self.params_dict['splitters']
            max_depths = self.params_dict['max_depths']
            min_samples_splits = self.params_dict['min_samples_splits']
            min_samples_leaves = self.params_dict['min_samples_leaves']
            for criterion, splitter, max_depth, min_samples_split, min_samples_leaf in tqdm(itertools.product(criteria, splitters, max_depths, min_samples_splits,min_samples_leaves)):
                accuracy = 0
                current_tree = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth,
                                                      min_samples_split=min_samples_split,
                                                      min_samples_leaf=min_samples_leaf)

                for train_index, test_index in skf.split(X_train, y_train):
                    X_tr, X_val = X_train[train_index], X_train[test_index]
                    y_tr, y_val = y_train[train_index], y_train[test_index]
                    current_tree.fit(X_tr, y_tr)
                    y_pred = current_tree.predict(X_val)
                    accuracy += accuracy_score(y_val, y_pred)
                accuracy /= cv
                if accuracy > self.best_accuracy:
                    self.best_model = current_tree
                    self.best_accuracy = accuracy
        self.best_model.fit(X_train, y_train)




data = pd.read_csv("../../data/clickbait_data.csv", sep="\t", header=None)
X = data[0]
y = data[1]
tokenizer = Tokenizer("wordpunct", True)
X = tokenizer.fit(X)
stopword = StopwordRemoval("english")
X = stopword.fit(X)
stemmer = Stemmer("english", "wordnet")
X = stemmer.fit(X)
vectorizer = Vectorizer("word2vec", "word2vec-google-news-300")
X = vectorizer.fit(X)
trainer = ClassicalTrainer("svm", {'kernels':['rbf'], 'gammas':['auto'], 'Cs':[0.1, 0.01, 1], 'degrees':[2]}, "accuracy")
#trainer = ClassicalTrainer("decision_tree", {'criteria':['gini', 'entropy'], 'splitters':['random', 'best'], 'max_depths':[10,100,1000, None], 'min_samples_splits':[2,3,4], 'min_samples_leaves':[1,2,3]}, "accuracy")
trainer.compute_best_params(X, y, 5)
print(trainer.best_accuracy)
print(trainer.best_model)
"""


