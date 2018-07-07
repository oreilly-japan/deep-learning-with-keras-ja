# -*- coding: utf-8 -*-
from collections import Counter

import numpy as np

PAD = "_PAD_"
UNK = "_UNK_"


def tokenize(text):
    return text.split()


def decode_tsv(line):
    label, text = line.strip().split("\t")
    return text, int(label)


def load_texts_and_labels(file):
    with open(file) as f:
        for line in f:
            text, label = decode_tsv(line)
            yield text, label


def create_vocab(texts, n):
    freq = Counter()
    for text in texts:
        words = tokenize(text)
        freq.update(words)
    freq = freq.most_common(n)
    vocab = {w: i+2 for i, (w, _) in enumerate(freq)}
    vocab[PAD] = 0
    vocab[UNK] = 1

    return vocab


def get_max_seq_len(data):
    max_seq_len = 0
    for text, _ in data:
        words = tokenize(text)
        max_seq_len = max(max_seq_len, len(words))

    return max_seq_len


def load_glove_vectors(file):
    word2emb = {}
    with open(file) as f:
        for line in f:
            cols = line.strip().split()
            word = cols[0]
            embedding = np.array(cols[1:], dtype="float32")
            word2emb[word] = embedding

    return word2emb


def make_weight_matrix(word2index, word2emb, vocab_size, embed_size):
    embedding_weights = np.zeros((vocab_size, embed_size))
    for word, index in word2index.items():
        try:
            embedding_weights[index, :] = word2emb[word]
        except KeyError:
            pass

    return embedding_weights
