# -*- coding: utf-8 -*-
from __future__ import division, print_function
import os
import collections

import nltk
import numpy as np
from keras.callbacks import TensorBoard
from keras.layers import Dense, Dropout, Conv1D, Embedding, GlobalMaxPooling1D
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import codecs

np.random.seed(42)

INPUT_FILE = os.path.join(os.path.dirname(__file__),
                          "data/umich-sentiment-train.txt")
GLOVE_MODEL = os.path.join(os.path.dirname(__file__),
                           "./data/glove.6B.300d.txt")
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
VOCAB_SIZE = 5000
EMBED_SIZE = 300
NUM_FILTERS = 256
NUM_WORDS = 3
BATCH_SIZE = 64
NUM_EPOCHS = 10

counter = collections.Counter()
with codecs.open(INPUT_FILE, "r", encoding="utf-8") as fin:
    maxlen = 0
    for line in fin:
        _, sent = line.strip().split("\t")
        try:
            words = [x.lower() for x in nltk.word_tokenize(sent)]
        except LookupError:
            print("Englisth tokenize does not downloaded. So download it.")
            nltk.download("punkt")
            words = [x.lower() for x in nltk.word_tokenize(sent)]
        if len(words) > maxlen:
            maxlen = len(words)
        for word in words:
            counter[word] += 1

word2index = collections.defaultdict(int)
for wid, word in enumerate(counter.most_common(VOCAB_SIZE)):
    word2index[word[0]] = wid + 1
vocab_sz = len(word2index) + 1
index2word = {v: k for k, v in word2index.items()}

xs, ys = [], []
with codecs.open(INPUT_FILE, "r", encoding="utf-8") as fin:
    for line in fin:
        label, sent = line.strip().split("\t")
        ys.append(int(label))
        words = [x.lower() for x in nltk.word_tokenize(sent)]
        wids = [word2index[word] for word in words]
        xs.append(wids)

X = pad_sequences(xs, maxlen=maxlen)
Y = np_utils.to_categorical(ys)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3,
                                                random_state=42)
print(Xtrain.shape, Xtest.shape, Ytrain.shape, Ytest.shape)

# load GloVe vectors
word2emb = {}
with codecs.open(GLOVE_MODEL, "r", encoding="utf-8") as fglove:
    for line in fglove:
        cols = line.strip().split()
        word = cols[0]
        embedding = np.array(cols[1:], dtype="float32")
        word2emb[word] = embedding

embedding_weights = np.zeros((vocab_sz, EMBED_SIZE))
for word, index in word2index.items():
    try:
        embedding_weights[index, :] = word2emb[word]
    except KeyError:
        pass

model = Sequential()
model.add(Embedding(vocab_sz, EMBED_SIZE, input_length=maxlen,
                    weights=[embedding_weights],
                    trainable=True))
model.add(Dropout(0.2))
model.add(Conv1D(filters=NUM_FILTERS, kernel_size=NUM_WORDS,
                 activation="relu"))
model.add(GlobalMaxPooling1D())
model.add(Dense(2, activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy",
              metrics=["accuracy"])

history = model.fit(Xtrain, Ytrain, batch_size=BATCH_SIZE,
                    epochs=NUM_EPOCHS,
                    callbacks=[TensorBoard(LOG_DIR)],
                    validation_data=(Xtest, Ytest))              

# evaluate model
score = model.evaluate(Xtest, Ytest, verbose=1)
print("Test score: {:.3f}, accuracy: {:.3f}".format(score[0], score[1]))
