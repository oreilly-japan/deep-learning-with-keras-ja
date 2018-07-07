# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import collections

import nltk
import numpy as np
from gensim.models import KeyedVectors
from keras.callbacks import TensorBoard
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import codecs


np.random.seed(42)

INPUT_FILE = os.path.join(os.path.dirname(__file__),
                          "data/umich-sentiment-train.txt")
WORD2VEC_MODEL = os.path.join(os.path.dirname(__file__),
                              "data/GoogleNews-vectors-negative300.bin.gz")
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
VOCAB_SIZE = 5000
EMBED_SIZE = 300
BATCH_SIZE = 64
NUM_EPOCHS = 10

print("reading data...")
counter = collections.Counter()
with codecs.open(INPUT_FILE, "r", encoding="utf-8") as f:
    maxlen = 0
    for line in f:
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

print("creating vocabulary...")
word2index = collections.defaultdict(int)
for wid, word in enumerate(counter.most_common(VOCAB_SIZE)):
    word2index[word[0]] = wid + 1
vocab_sz = len(word2index) + 1
index2word = {v: k for k, v in word2index.items()}
index2word[0] = "_UNK_"

print("creating word sequences...")
ws, ys = [], []
with codecs.open(INPUT_FILE, "r", encoding="utf-8") as fin:
    for line in fin:
        label, sent = line.strip().split("\t")
        ys.append(int(label))
        words = [x.lower() for x in nltk.word_tokenize(sent)]
        wids = [word2index[word] for word in words]
        ws.append(wids)

W = pad_sequences(ws, maxlen=maxlen)
Y = np_utils.to_categorical(ys)

# load GloVe vectors
print("loading word2vec vectors...")
word2vec = KeyedVectors.load_word2vec_format(WORD2VEC_MODEL, binary=True)

print("transferring embeddings...")
X = np.zeros((W.shape[0], EMBED_SIZE))
for i in range(W.shape[0]):
    E = np.zeros((EMBED_SIZE, maxlen))
    words = [index2word[wid] for wid in W[i].tolist()]
    for j in range(maxlen):
        try:
            E[:, j] = word2vec[words[j]]
        except KeyError:
            pass
    X[i, :] = np.sum(E, axis=1)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3,
                                                random_state=42)
print(Xtrain.shape, Xtest.shape, Ytrain.shape, Ytest.shape)

model = Sequential()
model.add(Dense(32, input_dim=EMBED_SIZE, activation="relu"))
model.add(Dropout(0.2))
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
