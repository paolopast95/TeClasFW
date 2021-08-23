from nltk.stem import PorterStemmer, LancasterStemmer, WordNetLemmatizer
from nltk.stem.snowball import ItalianStemmer, FrenchStemmer, SpanishStemmer, EnglishStemmer

class Stemmer():
    def __init__(self, language='english', stemmer_name='porter'):
        self.stemmer_name = stemmer_name
        self.language = language
        if language != "english" and stemmer_name != 'snowball':
            print(stemmer_name.title() + " does not exist for the " + language + " language!")
            print("Language changed to english..")
        if stemmer_name == "porter":
            self.stemmer = PorterStemmer()
        if stemmer_name == 'lancaster':
            self.stemmer = LancasterStemmer()
        if stemmer_name == 'wordnet':
            self.stemmer = WordNetLemmatizer()
        if stemmer_name == 'snowball':
            if language == "italian":
                self.stemmer = ItalianStemmer()
            if language == "french":
                self.stemmer = FrenchStemmer()
            if language == "spanish":
                self.stemmer = SpanishStemmer()
            if language == 'english':
                self.stemmer = EnglishStemmer()


    def fit(self, X):
        if self.stemmer_name != "wordnet":
            stemmed_X = [[self.stemmer.stem(word) for word in sentence] for sentence in X]
        else:
            stemmed_X = [[self.stemmer.lemmatize(word) for word in sentence] for sentence in X]
        return stemmed_X
