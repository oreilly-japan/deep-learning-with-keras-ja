# -*- coding: utf-8 -*-
from __future__ import division, print_function
import collections
import os

import nltk
import numpy as np
from keras.callbacks import TensorBoard
from keras.layers import Activation, Dense, Dropout, Embedding, LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import codecs


DATA_DIR = "./data"
LOG_DIR = "./logs"

MAX_FEATURES = 2000
MAX_SENTENCE_LENGTH = 40

EMBEDDING_SIZE = 128
HIDDEN_LAYER_SIZE = 64
BATCH_SIZE = 32
NUM_EPOCHS = 10

# Read training data and generate vocabulary
maxlen = 0
word_freqs = collections.Counter()
num_recs = 0
with codecs.open(os.path.join(DATA_DIR, "umich-sentiment-train.txt"), "r",
                 'utf-8') as ftrain:
    for line in ftrain:
        label, sentence = line.strip().split("\t")
        try:
            words = nltk.word_tokenize(sentence.lower())
        except LookupError:
            print("Englisth tokenize does not downloaded. So download it.")
            nltk.download("punkt")
            words = nltk.word_tokenize(sentence.lower())
        maxlen = max(maxlen, len(words))
        for word in words:
            word_freqs[word] += 1
        num_recs += 1

# Get some information about our corpus
print(maxlen)            # 42
print(len(word_freqs))   # 2313

# 1 is UNK, 0 is PAD
# We take MAX_FEATURES-1 features to account for PAD
vocab_size = min(MAX_FEATURES, len(word_freqs)) + 2
word2index = {x[0]: i+2 for i, x in
              enumerate(word_freqs.most_common(MAX_FEATURES))}
word2index["PAD"] = 0
word2index["UNK"] = 1
index2word = {v: k for k, v in word2index.items()}

# convert sentences to sequences
X = np.empty((num_recs, ), dtype=list)
y = np.zeros((num_recs, ))
i = 0
with codecs.open(os.path.join(DATA_DIR, "umich-sentiment-train.txt"),
                 'r', 'utf-8') as ftrain:
    for line in ftrain:
        label, sentence = line.strip().split("\t")
        words = nltk.word_tokenize(sentence.lower())
        seqs = []
        for word in words:
            if word in word2index:
                seqs.append(word2index[word])
            else:
                seqs.append(word2index["UNK"])
        X[i] = seqs
        y[i] = int(label)
        i += 1

# Pad the sequences (left padded with zeros)
X = sequence.pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH)

# Split input into training and test
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2,
                                                random_state=42)
print(Xtrain.shape, Xtest.shape, ytrain.shape, ytest.shape)

# Build model
model = Sequential()
model.add(Embedding(vocab_size, EMBEDDING_SIZE,
                    input_length=MAX_SENTENCE_LENGTH))
model.add(Dropout(0.5))
model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.5, recurrent_dropout=0.5))
model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam",
              metrics=["accuracy"])


history = model.fit(Xtrain, ytrain, batch_size=BATCH_SIZE,
                    epochs=NUM_EPOCHS,
                    callbacks=[TensorBoard(LOG_DIR)],
                    validation_data=(Xtest, ytest))

# evaluate
score, acc = model.evaluate(Xtest, ytest, batch_size=BATCH_SIZE)
print("Test score: {:.3f}, accuracy: {:.3f}".format(score, acc))

for i in range(5):
    idx = np.random.randint(len(Xtest))
    xtest = Xtest[idx].reshape(1, 40)
    ylabel = ytest[idx]
    ypred = model.predict(xtest)[0][0]
    sent = " ".join([index2word[x] for x in xtest[0].tolist() if x != 0])
    print("{:.0f}\t{:.0f}\t{}".format(ypred, ylabel, sent))
