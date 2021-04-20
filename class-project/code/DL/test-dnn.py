from collections import Counter
from torch.autograd import Variable

import torch
import torch.nn as nn
import numpy as np
import text_preprocessing as txt

import text_preprocessing as txt
import numpy as np

# # working
# from keras.datasets import reuters
# # this depends on machine computation capacity
# MAX_NB_WORDS = 7500
#
# print("Load Reuters dataset....")
# (X_train, y_train), (X_test, y_test) = reuters.load_data(num_words=MAX_NB_WORDS)
# word_index = reuters.get_word_index()
# index_word = {v: k for k, v in word_index.items()}
# X_train = [txt.text_cleaner(' '.join(index_word.get(w) for w in x)) for x in X_train]
# X_test = [txt.text_cleaner(' '.join(index_word.get(w) for w in x)) for x in X_test]
# X_train = np.array(X_train)
# X_train = np.array(X_train).ravel()
# print(X_train.shape)
# X_test = np.array(X_test)
# X_test = np.array(X_test).ravel()
# num_classes = 46

# working
from keras.datasets import imdb

# this depends on machine computation capacity
MAX_NB_WORDS = 75000

print("Load IMDB dataset....")
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=MAX_NB_WORDS)
word_index = imdb.get_word_index()
index_word = {v: k for k, v in word_index.items()}
X_train = [txt.text_cleaner(' '.join(index_word.get(w) for w in x)) for x in X_train]
X_test = [txt.text_cleaner(' '.join(index_word.get(w) for w in x)) for x in X_test]
X_train = np.array(X_train)
X_train = np.array(X_train).ravel()
print(X_train.shape)
X_test = np.array(X_test)
X_test = np.array(X_test).ravel()
num_classes = 2

# # working
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
#
# num_classes = 20

SEED = 1234

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

vocab = Counter()

for text in X_train:
    for word in text.split(' '):
        vocab[word.lower()] += 1

for text in X_test:
    for word in text.split(' '):
        vocab[word.lower()] += 1

total_words = len(vocab)


def get_word_2_index(vocab):
    word2index = {}
    for i, word in enumerate(vocab):
        word2index[word.lower()] = i

    return word2index


word2index = get_word_2_index(vocab)


def get_batch1(X_train, y_train, i, batch_size):
    batches = []
    results = []
    texts = X_train[i * batch_size:i * batch_size + batch_size]
    categories = y_train[i * batch_size:i * batch_size + batch_size]
    for text in texts:
        layer = np.zeros(total_words, dtype=float)
        for word in text.split(' '):
            layer[word2index[word.lower()]] += 1
        batches.append(layer)

    for category in categories:
        results.append(category)

    return np.array(batches), np.array(results)


# Parameters
learning_rate = 0.01
num_epochs = 10
batch_size = 150
display_step = 1

# Network Parameters
hidden_size = 100
input_size = total_words


class SimpleDNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleDNN, self).__init__()
        self.layer_1 = nn.Linear(input_size, hidden_size, bias=True)
        self.relu_1 = nn.ReLU()
        self.layer_2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.relu_2 = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size, num_classes, bias=True)

    def forward(self, x):
        out = self.layer_1(x)
        out = self.relu_1(out)
        out = self.layer_2(out)
        out = self.relu_2(out)
        out = self.output_layer(out)
        return out


loss = nn.CrossEntropyLoss()
input = Variable(torch.randn(2, 5), requires_grad=True)
target = Variable(torch.LongTensor(2).random_(5))
output = loss(input, target)
output.backward()

net = SimpleDNN(input_size, hidden_size, num_classes)

print(net)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# Train the Model
for epoch in range(num_epochs):
    total_batch = int(len(X_train) / batch_size)
    epoch_loss = 0
    for i in range(total_batch):
        batch_x, batch_y = get_batch1(X_train, y_train, i, batch_size)
        articles = Variable(torch.FloatTensor(batch_x))
        labels = Variable(torch.LongTensor(batch_y))
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(articles)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print("Loss:", epoch_loss)

# Test the Model
correct = 0
total = 0
total_test_data = len(y_test)
batch_x_test, batch_y_test = get_batch1(X_test, y_test, 0, total_test_data)
articles = Variable(torch.FloatTensor(batch_x_test))
labels = torch.LongTensor(batch_y_test)
outputs = net(articles)
_, predicted = torch.max(outputs.data, 1)
total += labels.size(0)
correct += (predicted == labels).sum()

print('Accuracy of the network on the 1180 test articles: %d %%' % (100 * correct / total))

from sklearn import metrics

print(metrics.classification_report(y_test, predicted))
print("Accuracy", metrics.accuracy_score(y_test, predicted))
print("F1-score", metrics.f1_score(y_test, predicted, average='weighted'))
