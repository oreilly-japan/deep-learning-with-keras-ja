import os
import tarfile
from collections import Counter
from urllib.request import urlretrieve
import numpy as np
from keras.layers import Input, add, concatenate, dot
from keras.layers.core import Activation, Dense, Dropout, Permute
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.preprocessing.text import text_to_word_sequence
from keras.utils import to_categorical
from keras.callbacks import TensorBoard


class bAbI():

    def __init__(self, use_10k=True, data_root="", padding="PAD"):
        self.url = "http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz"  # noqa
        self.vocab = []
        self.story_size = -1
        self.question_size = -1
        self.data_root = data_root
        self.use_10k = use_10k
        if not self.data_root:
            self.data_root = os.path.join(os.path.dirname(__file__), "data")
        self.PAD = padding

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def data_dir(self):
        _dir = "tasks_1-20_v1-2/"
        _dir += "en-10k" if self.use_10k else "en"
        return _dir

    def _get_location(self, kind="train"):
        file_name = "qa1_single-supporting-fact_{}.txt".format(kind.lower())
        return self.data_dir + "/" + file_name

    def download(self):
        tar_file = os.path.basename(self.url)
        if os.path.exists(os.path.join(self.data_root, self.data_dir)):
            return
        if not os.path.exists(self.data_root):
            os.mkdir(self.data_root)

        file_path = os.path.join(self.data_root, tar_file)
        if not os.path.isfile(file_path):
            print("Download the bABI data...")
            urlretrieve(self.url, file_path)
        with tarfile.open(file_path, mode="r:gz") as gz:
            for kind in ["train", "test"]:
                target = self._get_location(kind)
                gz.extract(target, self.data_root)
        os.remove(file_path)

    def _read_qa(self, kind="train"):
        path = os.path.join(self.data_root, self._get_location(kind))
        stories, questions, answers = [], [], []
        with open(path, "r", encoding="utf-8") as f:
            story_lines = []
            for line in f:
                line = line.strip()
                index, text = line.split(" ", 1)
                if "\t" in text:
                    question, answer, _ = text.split("\t")
                    stories.append(" ".join(story_lines))
                    questions.append(question.strip())
                    answers.append(answer.strip())
                    story_lines = []
                else:
                    story_lines.append(text)

        return stories, questions, answers

    def make_vocab(self):
        train_s, train_q, train_a = self._read_qa(kind="train")
        test_s, test_q, test_a = self._read_qa(kind="test")

        all_s = train_s + test_s
        all_q = train_q + test_q

        # Make vocabulary from all stories and questions
        words = []
        for s, q in zip(all_s, all_q):
            s_words = self.tokenize(s)
            if len(s_words) > self.story_size:
                self.story_size = len(s_words)

            q_words = self.tokenize(q)
            if len(q_words) > self.question_size:
                self.question_size = len(q_words)

            words += s_words
            words += q_words

        word_count = Counter(words)
        words = [w_c[0] for w_c in word_count.most_common()]
        words.insert(0, self.PAD)  # add pad
        self.vocab = words

    def tokenize(self, string):
        words = text_to_word_sequence(string, lower=True)
        return words

    def get_batch(self, kind="train"):
        if self.vocab_size == 0:
            self.make_vocab()
        stories, questions, answers = self._read_qa(kind)
        s_indices = [self.to_indices(s, self.story_size) for s in stories]
        q_indices = [self.to_indices(q, self.question_size)
                     for q in questions]
        a_indices = [self.vocab.index(a) for a in answers]
        a_categorical = to_categorical(a_indices, num_classes=self.vocab_size)

        return np.array(s_indices), np.array(q_indices), a_categorical

    def to_indices(self, string, fit_length=-1):
        if self.vocab_size == 0:
            raise Exception("You have to execute make_vocab")
        words = self.tokenize(string)
        indices = [self.vocab.index(w) for w in words]
        if fit_length > 0:
            indices = indices[:fit_length]
            pad_size = fit_length - len(indices)
            if pad_size > 0:
                indices += [self.vocab.index(self.PAD)] * pad_size
        return indices

    def to_string(self, indices):
        words = [self.vocab[i] for i in indices]
        string = " ".join([w for w in words if w != self.PAD])
        return string


def make_model(story_size, question_size, vocab_size,
               embedding_size=64, latent_size=32, drop_rate=0.3):
    story_input = Input(shape=(story_size,))
    question_input = Input(shape=(question_size,))

    story_embed_for_a = Embedding(
                        input_dim=vocab_size,
                        output_dim=embedding_size,
                        input_length=story_size)
    question_embed = Embedding(
                        input_dim=vocab_size,
                        output_dim=embedding_size,
                        input_length=question_size)
    story_encoder_for_a = Dropout(drop_rate)(story_embed_for_a(story_input))
    question_encoder = Dropout(drop_rate)(question_embed(question_input))

    # match story & question along seq_size to make attention on story
    # (axes=[batch, seq_size, embed_size] after encoding)
    match = dot([story_encoder_for_a, question_encoder], axes=[2, 2])
    
    story_embed_for_c = Embedding(
        input_dim=vocab_size,
        output_dim=question_size,
        input_length=story_size
    )
    story_encoder_for_c = Dropout(drop_rate)(story_embed_for_c(story_input))

    # merge match and story context
    response = add([match, story_encoder_for_c])
    # (question_size x story_size) => (story_size x question_size)
    response = Permute((2, 1))(response)

    answer = concatenate([response, question_encoder], axis=-1)
    answer = LSTM(latent_size)(answer)
    answer = Dropout(drop_rate)(answer)
    answer = Dense(vocab_size)(answer)
    output = Activation("softmax")(answer)
    model = Model(inputs=[story_input, question_input], outputs=output)

    return model


def main(batch_size, epochs, show_result_count):
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    corpus = bAbI()
    corpus.download()
    corpus.make_vocab()
    train_s, train_q, train_a = corpus.get_batch(kind="train")
    test_s, test_q, test_a = corpus.get_batch(kind="test")
    print("{} train data, {} test data.".format(len(train_s), len(test_s)))
    print("vocab size is {}.".format(corpus.vocab_size))

    model = make_model(
                corpus.story_size, corpus.question_size, corpus.vocab_size)

    # train the model
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit([train_s, train_q], [train_a],
              validation_data=([test_s, test_q], [test_a]),
              batch_size=batch_size, epochs=epochs,
              callbacks=[TensorBoard(log_dir=log_dir)]
              )

    answer = np.argmax(test_a, axis=1)
    predicted = model.predict([test_s, test_q])
    predicted = np.argmax(predicted, axis=1)

    for i in range(show_result_count):
        story = corpus.to_string(test_s[i].tolist())
        question = corpus.to_string(test_q[i].tolist())
        a = corpus.to_string([answer[i]])
        p = corpus.to_string([predicted[i]])
        ox = "o" if a == p else "x"
        print(story + "\n", question + "\n",
              "{} True: {}, Predicted: {}".format(ox, a, p))


if __name__ == "__main__":
    main(batch_size=64, epochs=50, show_result_count=10)
