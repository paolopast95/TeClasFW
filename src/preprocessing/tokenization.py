from nltk.tokenize import word_tokenize, wordpunct_tokenize, WhitespaceTokenizer
import re

class Tokenizer():
    def __init__(self, type, remove_punctuation = False):
        self.remove_punctuation = remove_punctuation
        self.type = type

    def fit(self, X):
        if self.remove_punctuation:
            tokenized_X = [re.sub(r'[^\w\s]', '', s).lower() for s in X]
        else:
            tokenized_X = [sentence.lower() for sentence in X]
        if self.type == 'whitespace':
            tokenizer = WhitespaceTokenizer()
            tokenized_X = [tokenizer.tokenize(sentence) for sentence in tokenized_X]
        elif self.type == 'word':
            tokenized_X = [word_tokenize(sentence) for sentence in tokenized_X]
        elif self.type == 'wordpunct':
            tokenized_X = [wordpunct_tokenize(sentence) for sentence in tokenized_X]
        else:
            print(self.type + " tokenizer does not exists")
            tokenized_X = []
        return tokenized_X



