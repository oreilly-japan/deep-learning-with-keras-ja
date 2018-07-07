import numpy as np
from keras import backend as K
from keras.layers import Input, Lambda
from keras.models import Model


def euclidean_distance(vecs):
    x, y = vecs
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def euclidean_distance_output_shape(input_shapes):
    shape1, shape2 = input_shapes
    assert shape1 == shape2  # shape have to be equal
    return (shape1[0], 1)


def measure_model(row_count=4, vec_size=3):
    left = Input(shape=(vec_size,))
    right = Input(shape=(vec_size,))

    distance = Lambda(euclidean_distance,
                      output_shape=euclidean_distance_output_shape
                      )([left, right])
    model = Model([left, right], distance)

    size = row_count * vec_size
    left_mx = np.random.randint(9, size=size).reshape((row_count, vec_size))
    right_mx = np.random.randint(9, size=size).reshape((row_count, vec_size))

    output = model.predict([left_mx, right_mx])
    print("Distance between\n {} \nand\n {} \nis\n {}".format(
        left_mx, right_mx, output
    ))


if __name__ == "__main__":
    measure_model()

