import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import re
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download("stopwords")
cachedStopWords = stopwords.words("english")


def transliterate(line):
    cedilla2latin = [[u'Á', u'A'], [u'á', u'a'], [u'Č', u'C'], [u'č', u'c'], [u'Š', u'S'], [u'š', u's']]
    tr = dict([(a[0], a[1]) for (a) in cedilla2latin])
    new_line = ""
    for letter in line:
        if letter in tr:
            new_line += tr[letter]
        else:
            new_line += letter
    return new_line


def text_cleaner(text,
                 deep_clean=False,
                 stem=True,
                 stop_words=True,
                 translite_rate=True):
    rules = [
        {r'>\s+': u'>'},  # remove spaces after a tag opens or closes
        {r'\s+': u' '},  # replace consecutive spaces
        {r'\s*<br\s*/?>\s*': u'\n'},  # newline after a <br>
        {r'</(div)\s*>\s*': u'\n'},  # newline after </p> and </div> and <h1/>...
        {r'</(p|h\d)\s*>\s*': u'\n\n'},  # newline after </p> and </div> and <h1/>...
        {r'<head>.*<\s*(/head|body)[^>]*>': u''},  # remove <head> to </head>
        {r'<a\s+href="([^"]+)"[^>]*>.*</a>': r'\1'},  # show links instead of texts
        {r'[ \t]*<[^<]*?/?>': u''},  # remove remaining tags
        {r'^\s+': u''}  # remove spaces at the beginning

    ]

    if deep_clean:
        text = text.replace(".", "")
        text = text.replace("[", " ")
        text = text.replace(",", " ")
        text = text.replace("]", " ")
        text = text.replace("(", " ")
        text = text.replace(")", " ")
        text = text.replace("\"", "")
        text = text.replace("-", " ")
        text = text.replace("=", " ")
        text = text.replace("?", " ")
        text = text.replace("!", " ")

        for rule in rules:
            for (k, v) in rule.items():
                regex = re.compile(k)
                text = regex.sub(v, text)
            text = text.rstrip()
            text = text.strip()
        text = text.replace('+', ' ').replace('.', ' ').replace(',', ' ').replace(':', ' ')
        text = re.sub("(^|\W)\d+($|\W)", " ", text)
        if translite_rate:
            text = transliterate(text)
        if stem:
            text = PorterStemmer().stem(text)
        text = WordNetLemmatizer().lemmatize(text)
        if stop_words:
            stop_words = set(stopwords.words('english'))
            word_tokens = word_tokenize(text)
            text = [w for w in word_tokens if not w in stop_words]
            text = ' '.join(str(e) for e in text)
    else:
        for rule in rules:
            for (k, v) in rule.items():
                regex = re.compile(k)
                text = regex.sub(v, text)
            text = text.rstrip()
            text = text.strip()
    return text.lower()
