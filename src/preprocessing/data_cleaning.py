import re
import dateutil



class DataCleaner():
    def __init__(self, cleaning_conf):
        self.lowercase = cleaning_conf['lowercase']
        self.urls = cleaning_conf['replace_urls']
        self.numbers = cleaning_conf['remove_numbers']
        self.emoji = cleaning_conf['replace_emoji']
        self.punctuation = cleaning_conf['remove_punctuation']
        self.citations = cleaning_conf['replace_citations']
        self.hashtags = cleaning_conf['remove_hashtags']



    def transform(self, X):
        cleaned_X = X
        if self.lowercase:
            cleaned_X = [sentence.lower() for sentence in cleaned_X]
        if not self.citations is None:
            cleaned_X = [re.sub("@[A-Za-z0-9]+",self.citations,sentence)for sentence in cleaned_X]
        if not self.urls is None:
            cleaned_X = [re.sub(r"(?:\@|http?\://|https?\://|www)\S+", self.urls, sentence) for sentence in cleaned_X]
        if self.hashtags:
            cleaned_X = [' '.join(el.replace("#", "") for el in re.split('\#[\w|_]+]', sentence)) for sentence in cleaned_X]
        if not self.emoji is None:
            emoji_pattern = re.compile("["
                                       u"\U0001F600-\U0001F64F"  # emoticons
                                       u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                       u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                       u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                       u"\U00002702-\U000027B0"
                                       u"\U000024C2-\U0001F251"
                                       "]+", flags=re.UNICODE)
            cleaned_X = [emoji_pattern.sub(self.emoji, sentence) for sentence in cleaned_X]
            if self.hashtags:
                cleaned_X = [sentence.replace("#", "").replace("_", " ") for sentence in cleaned_X]
            if self.numbers:
                cleaned_X = [re.sub(r'[0-9]+', '', sentence) for sentence in cleaned_X]
        cleaned_X = [re.sub(' +', ' ', sentence)for sentence in cleaned_X]
        return cleaned_X


dc = DataCleaner(lowercase=True, hashtags=True, urls="<URL>", emoji="<EMOJI>", numbers=True)
X = dc.transform(["ciao #paolo_past 20 Jul 2011 https://stackoverflow.com/questions/51384426/remove-recognized-date-from-string 🎸   "])
print(X)