import nltk
from nltk.corpus import stopwords

class StopwordRemoval():
    def __init__(self, language):
        nltk.download("stopwords")
        self.language = language
        self.stopwords = stopwords.words(language)

    def fit(self, X):
        filtered_X = [[word for word in sentence if word not in self.stopwords] for sentence in X]
        return filtered_X
