from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import gensim
import gensim.downloader as gensim_api
import pickle
import os
import numpy as np

class Vectorizer:
    def __init__(self, operation_type="count", embedding_filename=None, max_features=5000):
        self.operation_type = operation_type
        self.embedding_filename = embedding_filename
        self.max_features = max_features

    def fit(self, X):
        X_str = [" ".join(sent) for sent in X]
        if self.operation_type=="count":
            vectorizer = CountVectorizer(max_features=self.max_features)
            X_vect = vectorizer.fit_transform(X_str)
        elif self.operation_type == "tfidf":
            vectorizer = TfidfVectorizer(max_features=self.max_features)
            X_vect = vectorizer.fit_transform(X_str)
        elif self.operation_type == "word2vec":
            if os.path.exists(os.path.join("../../embeddings/",self.embedding_filename)):
                with open(os.path.join("../../embeddings/",self.embedding_filename), "rb") as f:
                    nlp = pickle.load(f)
            else:
                nlp = gensim_api.load("word2vec-google-news-300")
                with open(os.path.join("../../embeddings/",self.embedding_filename), "wb") as f:
                    pickle.dump(nlp, f)
            X_vect = [np.mean([nlp[word] for word in sentence if word in nlp] or [np.zeros(300)], axis=0) for sentence in X]
        return X_vect




v = Vectorizer("word2vec", "word2vec-google-news-300")
v.fit([["ciao","mi","chiamo"],["ciao", "mi", "piace", "la", "pizza", "pizza"]])