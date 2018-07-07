# -*- coding: utf-8 -*-
from __future__ import division, print_function
import collections
import os

import nltk
import numpy as np
from keras.layers import Activation, Dense, Dropout, RepeatVector, Embedding, \
    GRU, LSTM, TimeDistributed, Bidirectional
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.utils import np_utils
from sklearn.model_selection import train_test_split


def parse_sentences(filename):
    word_freqs = collections.Counter()
    num_recs, maxlen = 0, 0
    with open(filename, "r") as fin:
        for line in fin:
            words = line.strip().lower().split()
            for word in words:
                word_freqs[word] += 1
            maxlen = max(maxlen, len(words))
            num_recs += 1
    return word_freqs, maxlen, num_recs


def build_tensor(filename, numrecs, word2index, maxlen):
    data = np.empty((numrecs, ), dtype=list)
    with open(filename, "r") as fin:
        for i, line in enumerate(fin):
            wids = []
            for word in line.strip().lower().split():
                if word in word2index:
                    wids.append(word2index[word])
                else:
                    wids.append(word2index["UNK"])
            data[i] = wids
    pdata = sequence.pad_sequences(data, maxlen=maxlen)
    return pdata


DATA_DIR = "./data"

with open(os.path.join(DATA_DIR, "treebank_sents.txt"), "w") as fedata, \
        open(os.path.join(DATA_DIR, "treebank_poss.txt"), "w") as ffdata:
    sents = nltk.corpus.treebank.tagged_sents()
    for sent in sents:
        words, poss = [], []
        for word, pos in sent:
            if pos == "-NONE-":
                continue
            words.append(word)
            poss.append(pos)
        fedata.write("{:s}\n".format(" ".join(words)))
        ffdata.write("{:s}\n".format(" ".join(poss)))


s_wordfreqs, s_maxlen, s_numrecs = \
    parse_sentences(os.path.join(DATA_DIR, "treebank_sents.txt"))
t_wordfreqs, t_maxlen, t_numrecs = \
    parse_sentences(os.path.join(DATA_DIR, "treebank_poss.txt"))
print("# records: {:d}".format(s_numrecs))
print("# unique words: {:d}".format(len(s_wordfreqs)))
print("# unique POS tags: {:d}".format(len(t_wordfreqs)))
print("# words/sentence: max: {:d}".format(s_maxlen))


MAX_SEQLEN = 250
S_MAX_FEATURES = 5000
T_MAX_FEATURES = 45


s_vocabsize = min(len(s_wordfreqs), S_MAX_FEATURES) + 2
s_word2index = {x[0]: i+2 for i, x in
                enumerate(s_wordfreqs.most_common(S_MAX_FEATURES))}
s_word2index["PAD"] = 0
s_word2index["UNK"] = 1
s_index2word = {v: k for k, v in s_word2index.items()}

t_vocabsize = len(t_wordfreqs) + 1
t_word2index = {x[0]: i for i, x in
                enumerate(t_wordfreqs.most_common(T_MAX_FEATURES))}
t_word2index["PAD"] = 0
t_index2word = {v: k for k, v in t_word2index.items()}


X = build_tensor(os.path.join(DATA_DIR, "treebank_sents.txt"),
                 s_numrecs, s_word2index, MAX_SEQLEN)
Y = build_tensor(os.path.join(DATA_DIR, "treebank_poss.txt"),
                 t_numrecs, t_word2index, MAX_SEQLEN)
Y = np.array([np_utils.to_categorical(d, t_vocabsize) for d in Y])
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y,
                                                test_size=0.2, random_state=42)


EMBED_SIZE = 128
HIDDEN_SIZE = 64
BATCH_SIZE = 32
NUM_EPOCHS = 1

# GRU
model = Sequential()
model.add(Embedding(s_vocabsize, EMBED_SIZE, input_length=MAX_SEQLEN))
model.add(Dropout(0.2))
model.add(GRU(HIDDEN_SIZE, dropout=0.2, recurrent_dropout=0.2))
model.add(RepeatVector(MAX_SEQLEN))
model.add(GRU(HIDDEN_SIZE, return_sequences=True))
model.add(TimeDistributed(Dense(t_vocabsize)))
model.add(Activation("softmax"))
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])
model.fit(Xtrain, Ytrain, batch_size=BATCH_SIZE,
          epochs=NUM_EPOCHS, validation_data=[Xtest, Ytest])
score, acc = model.evaluate(Xtest, Ytest, batch_size=BATCH_SIZE)
print("Test score: {:.3f}, accuracy: {:.3f}".format(score, acc))


# LSTM
model = Sequential()
model.add(Embedding(s_vocabsize, EMBED_SIZE, input_length=MAX_SEQLEN))
model.add(Dropout(0.2))
model.add(LSTM(HIDDEN_SIZE, dropout=0.2, recurrent_dropout=0.2))
model.add(RepeatVector(MAX_SEQLEN))
model.add(LSTM(HIDDEN_SIZE, return_sequences=True))
model.add(TimeDistributed(Dense(t_vocabsize)))
model.add(Activation("softmax"))
model.compile(loss="categorical_crossentropy",
              optimizer="adam", metrics=["accuracy"])
model.fit(Xtrain, Ytrain, batch_size=BATCH_SIZE,
          epochs=NUM_EPOCHS, validation_data=[Xtest, Ytest])
score, acc = model.evaluate(Xtest, Ytest, batch_size=BATCH_SIZE)
print("Test score: {:.3f}, accuracy: {:.3f}".format(score, acc))


# Bidirectional LSTM
model = Sequential()
model.add(Embedding(s_vocabsize, EMBED_SIZE, input_length=MAX_SEQLEN))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(HIDDEN_SIZE, dropout=0.2, recurrent_dropout=0.2)))
model.add(RepeatVector(MAX_SEQLEN))
model.add(Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True)))
model.add(TimeDistributed(Dense(t_vocabsize)))
model.add(Activation("softmax"))
model.compile(loss="categorical_crossentropy",
              optimizer="adam", metrics=["accuracy"])
model.fit(Xtrain, Ytrain, batch_size=BATCH_SIZE,
          epochs=NUM_EPOCHS, validation_data=[Xtest, Ytest])
score, acc = model.evaluate(Xtest, Ytest, batch_size=BATCH_SIZE)
print("Test score: {:.3f}, accuracy: {:.3f}".format(score, acc))
