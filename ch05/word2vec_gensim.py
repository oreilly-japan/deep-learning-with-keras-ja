# -*- coding: utf-8 -*-
from __future__ import print_function
import logging
import os
from gensim.models import word2vec


logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s",
                    level=logging.INFO)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
sentences = word2vec.Text8Corpus(os.path.join(DATA_DIR, "text8"), 50)
model = word2vec.Word2Vec(sentences, size=300, min_count=30)

print("model.most_similar('woman')")
print(model.most_similar("woman"))


print("model.most_similar(positive=['woman', 'king'], negative=['man'], topn=10)")
print(model.most_similar(positive=["woman", "king"],
                         negative=["man"],
                         topn=10))

print("model.similarity('girl', 'woman')")
print(model.similarity("girl", "woman"))
print("model.similarity('girl', 'man')")
print(model.similarity("girl", "man"))
print("model.similarity('girl', 'car')")
print(model.similarity("girl", "car"))
print("model.similarity('bus', 'car')")
print(model.similarity("bus", "car"))
