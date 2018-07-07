from keras.layers import Dense, Dropout, Conv1D, Embedding, GlobalMaxPooling1D
from keras.models import Sequential


def build_sentiment_model(vocab_size, embed_size, maxlen,
                          num_filters, num_words, embedding_weights=None):
    model = Sequential()
    if embedding_weights is None:
        model.add(Embedding(vocab_size, embed_size, input_length=maxlen))
    else:
        model.add(Embedding(vocab_size, embed_size,
                            input_length=maxlen,
                            weights=[embedding_weights]))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters=num_filters, kernel_size=num_words,
                     activation="relu"))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(2, activation="softmax"))

    return model
