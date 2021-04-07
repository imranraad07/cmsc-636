from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import text_preprocessing as txt
import numpy as np

# need a gpu machine to run this
# from sklearn.datasets import fetch_20newsgroups
#
# newsgroups_train = fetch_20newsgroups(subset='train')
# newsgroups_test = fetch_20newsgroups(subset='test')
# X_train = newsgroups_train.data
# X_test = newsgroups_test.data
# y_train = newsgroups_train.target
# y_test = newsgroups_test.target
#
# X_train = [txt.text_cleaner(x) for x in X_train]
# X_test = [txt.text_cleaner(x) for x in X_test]
# X_train = np.array(X_train)
# X_train = np.array(X_train).ravel()
# print(X_train.shape)
# X_test = np.array(X_test)
# X_test = np.array(X_test).ravel()

# # this depends on machine computation capacity
MAX_NB_WORDS = 75000

from keras.datasets import imdb
print("Load IMDB dataset....")
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=MAX_NB_WORDS)
word_index = imdb.get_word_index()

# from keras.datasets import reuters
#
# print("Load Reuters dataset....")
# (X_train, y_train), (X_test, y_test) = reuters.load_data(num_words=MAX_NB_WORDS)
# word_index = reuters.get_word_index()

index_word = {v: k for k, v in word_index.items()}
X_train = [txt.text_cleaner(' '.join(index_word.get(w) for w in x)) for x in X_train]
X_test = [txt.text_cleaner(' '.join(index_word.get(w) for w in x)) for x in X_test]
X_train = np.array(X_train)
X_train = np.array(X_train).ravel()
print(X_train.shape)
X_test = np.array(X_test)
X_test = np.array(X_test).ravel()

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', BaggingClassifier(KNeighborsClassifier())),
                     ])

text_clf.fit(X_train, y_train)

predicted = text_clf.predict(X_test)

print(metrics.classification_report(y_test, predicted))
print("Accuracy", metrics.accuracy_score(y_test, predicted))
print("F1-score", metrics.f1_score(y_test, predicted, average='weighted'))
