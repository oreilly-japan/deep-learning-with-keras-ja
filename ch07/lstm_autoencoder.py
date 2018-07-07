import os
import argparse
from zipfile import ZipFile
from urllib.request import urlretrieve
from collections import Counter
import re
import numpy as np
import nltk
from nltk.corpus import reuters
from nltk.corpus import stopwords
from keras.layers import Input, LSTM, Bidirectional, RepeatVector
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import load_model


class ReutersCorpus():

    def __init__(self, padding="PAD", unknown="UNK"):
        self.documents = []
        self.stopwords = []
        self.vocab = []
        self._ignores = re.compile("[.,-/\"'>()&;:]")
        self.PAD = padding
        self.UNK = unknown
        try:
            self.documents = reuters.fileids()
        except LookupError:
            print("Reuters corpus does not downloaded. So download it.")
            nltk.download("reuters")
            self.documents = reuters.fileids()

        try:
            self.stopwords = stopwords.words("english")
        except LookupError:
            print("Englisth stopword does not downloaded. So download it.")
            nltk.download("stopwords")
            self.stopwords = stopwords.words("english")

    def build(self, vocab_size=5000):
        words = reuters.words()
        words = [self.trim(w) for w in words]
        words = [w for w in words if w]
        freq = Counter(words)
        freq = freq.most_common(vocab_size)
        self.vocab = [w_c[0] for w_c in freq]
        self.vocab = [self.PAD, self.UNK] + self.vocab

    def trim(self, word):
        w = word.lower().strip()
        if w in self.stopwords or self._ignores.match(w):
            return ""
        if w.replace(".", "").isdigit():
            return "9"
        return w

    def batch_iter(self, embedding, kind="train", batch_size=64, seq_size=50):
        if len(self.vocab) == 0:
            raise Exception(
                "Vocabulary hasn't made yet. Please execute 'build' method."
                )

        steps = self.get_step_count(kind, batch_size)
        docs = self.get_documents(kind)
        docs_i = self.docs_to_matrix(docs, seq_size)
        docs = None  # free memory

        while True:
            indices = np.random.permutation(np.arange(len(docs_i)))
            for s in range(steps):
                index = s * batch_size
                x = docs_i[indices[index:(index + batch_size)]]
                x_vec = embedding[x]
                # input = output
                yield x_vec, x_vec

    def docs_to_matrix(self, docs, seq_size):
        docs_i = []
        for d in docs:
            words = reuters.words(d)
            words = self.sentence_to_ids(words, seq_size)
            docs_i.append(words)
        docs_i = np.array(docs_i)
        return docs_i

    def sentence_to_ids(self, sentence, seq_size):
        v = self.vocab
        UNK = v.index(self.UNK)
        PAD = v.index(self.PAD)
        words = [self.trim(w) for w in sentence][:seq_size]
        words = [v.index(w) if w in v else UNK for w in words if w]
        if len(words) < seq_size:
            words += [PAD] * (seq_size - len(words))
        return words

    def get_step_count(self, kind="train", batch_size=64):
        size = len(self.get_documents(kind))
        return size // batch_size

    def get_documents(self, kind="train"):
        docs = list(filter(lambda doc: doc.startswith(kind), self.documents))
        return docs


class EmbeddingLoader():

    def __init__(self, embed_dir="", size=100):
        self.embed_dir = embed_dir
        self.size = size
        if not self.embed_dir:
            self.embed_dir = os.path.join(os.path.dirname(__file__), "embed")

    def load(self, seq_size, corpus, download=True):
        url = "http://nlp.stanford.edu/data/wordvecs/glove.6B.zip"
        embed_name = "glove.6B.{}d.txt".format(self.size)
        embed_path = os.path.join(self.embed_dir, embed_name)
        if not os.path.isfile(embed_path):
            if not download:
                raise Exception(
                    "Can't load embedding from {}.".format(embed_path)
                    )
            else:
                print("Download the GloVe embedding.")
                file_name = os.path.basename(url)
                if not os.path.isdir(self.embed_dir):
                    os.mkdir(self.embed_dir)
                zip_path = os.path.join(self.embed_dir, file_name)
                urlretrieve(url, zip_path)
                with ZipFile(zip_path) as z:
                    z.extractall(self.embed_dir)
                    os.remove(zip_path)

        vocab = corpus.vocab
        if len(vocab) == 0:
            raise Exception("You have to make vocab by 'build' method.")
        embed_matrix = np.zeros((len(vocab), self.size))
        UNK = vocab.index(corpus.UNK)
        with open(embed_path, mode="r", encoding="utf-8") as f:
            for line in f:
                values = line.strip().split()
                word = values[0].strip()
                vector = np.asarray(values[1:], dtype="float32")
                if word in vocab:
                    index = vocab.index(word)
                    embed_matrix[index] = vector
        embed_matrix[UNK] = np.random.uniform(-1, 1, self.size)
        return embed_matrix


class AutoEncoder():

    def __init__(self, seq_size=50, embed_size=100, latent_size=256):
        self.seq_size = seq_size
        self.embed_size = embed_size
        self.latent_size = latent_size
        self.model = None

    def build(self):
        inputs = Input(shape=(self.seq_size, self.embed_size), name="input")
        encoded = Bidirectional(
            LSTM(self.latent_size),
            merge_mode="concat", name="encoder")(inputs)
        encoded = RepeatVector(self.seq_size, name="replicate")(encoded)
        decoded = Bidirectional(
            LSTM(self.embed_size, return_sequences=True),
            merge_mode="sum", name="decoder")(encoded)

        self.model = Model(inputs, decoded)

    @classmethod
    def load(cls, path):
        model = load_model(path)
        _, seq_size, embed_size = model.input.shape  # top is batch size
        latent_size = model.get_layer("encoder").input_shape[1]
        ae = AutoEncoder(seq_size, embed_size, latent_size)
        ae.model = model
        return ae

    def get_encoder(self):
        if self.model:
            m = self.model
            encoder = Model(m.input, m.get_layer("encoder").output)
            return encoder
        else:
            raise Exception("Model is not built/loaded")


def main(log_dir, model_name="autoencoder.h5"):
    print("1. Prepare the corpus.")
    corpus = ReutersCorpus()
    corpus.build(vocab_size=5000)

    print("2. Make autoencoder model.")
    ae = AutoEncoder(seq_size=50, embed_size=100, latent_size=512)
    ae.build()

    print("3. Load GloVe embeddings.")
    embed_loader = EmbeddingLoader(size=ae.embed_size)
    embedding = embed_loader.load(ae.seq_size, corpus)

    print("4. Train the model (trained model is saved to {}).".format(log_dir))
    batch_size = 64
    ae.model.compile(optimizer="sgd", loss="mse")
    model_file = os.path.join(log_dir, model_name)
    train_iter = corpus.batch_iter(embedding, "train", batch_size, ae.seq_size)
    test_iter = corpus.batch_iter(embedding, "test", batch_size, ae.seq_size)
    train_steps = corpus.get_step_count("train", batch_size)
    test_steps = corpus.get_step_count("test", batch_size)

    ae.model.fit_generator(
        train_iter, train_steps,
        epochs=20,
        validation_data=test_iter,
        validation_steps=test_steps,
        callbacks=[
            TensorBoard(log_dir=log_dir), 
            ModelCheckpoint(filepath=model_file, save_best_only=True)
            ]
        )


def predict(log_dir, model_name="autoencoder.h5"):
    print("1. Load the trained model.")
    model_file = os.path.join(log_dir, model_name)
    ae = AutoEncoder.load(model_file)

    print("2. Prepare the corpus.")
    corpus = ReutersCorpus()
    test_docs = corpus.get_documents("test")
    labels = [reuters.categories(f)[0] for f in test_docs]
    categories = Counter(labels).most_common()
    # Use categories that has more than 30 documents
    categories = [c[0] for c in categories if c[1] > 50]
    filtered = [i for i, lb in enumerate(labels) if lb in categories]
    labels = [categories.index(labels[i]) for i in filtered]
    test_docs = [test_docs[i] for i in filtered]
    corpus.build(vocab_size=5000)

    print("3. Load GloVe embeddings.")
    embed_loader = EmbeddingLoader(size=ae.embed_size)
    embedding = embed_loader.load(ae.seq_size, corpus)

    print("4. Use model's encoder to classify the documents.")
    from sklearn.cluster import KMeans
    docs = corpus.docs_to_matrix(test_docs, ae.seq_size)
    doc_vecs = embedding[docs]
    features = ae.get_encoder().predict(doc_vecs)
    clf = KMeans(n_clusters=len(categories))
    clf.fit(features)
    ae_dist = clf.inertia_

    from sklearn.feature_extraction.text import CountVectorizer
    test_doc_words = [" ".join(reuters.words(d)) for d in test_docs]
    vectorizer = CountVectorizer(vocabulary=corpus.vocab)
    c_features = vectorizer.fit_transform(test_doc_words)
    clf.fit(c_features)
    cnt_dist = clf.inertia_
    print(" Sum of distances^2 of samples to their closest center is")
    print(" Autoencoder: {}".format(ae_dist))
    print(" Word count base: {}".format(cnt_dist))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Try text autoencoder by reuters corpus")
    parser.add_argument(
        "--predict", action="store_const", const=True, default=False,
        help="Classify the sentences by trained model")

    args = parser.parse_args()
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    if args.predict:
        predict(log_dir)
    else:
        main(log_dir)
