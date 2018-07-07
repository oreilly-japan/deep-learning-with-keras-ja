from keras import backend as K
from keras.engine.topology import Layer
from keras.layers.core import Dropout, Reshape
from keras.layers.convolutional import ZeroPadding2D
from keras.models import Sequential
import numpy as np


def test_layer(layer, x):
    # Adjust layer input_shape to x.shape
    layer_config = layer.get_config()
    layer_config["input_shape"] = x.shape
    layer = layer.__class__.from_config(layer_config)
    model = Sequential()
    model.add(layer)
    # 1. Test building the computation graph process
    model.compile("rmsprop", "mse")
    _x = np.expand_dims(x, axis=0)  # Add dimension for batch size

    # 2. Test run the graph process
    return model.predict(_x)[0]


class LocalResponseNormalization(Layer):

    def __init__(self, n=5, alpha=0.0005, beta=0.75, k=2, **kwargs):
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.k = k
        super(LocalResponseNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        # In this layer, no trainable weight is used.
        super(LocalResponseNormalization, self).build(input_shape)

    def call(self, x):
        squared = K.square(x)
        # WITHIN_CHANNEL Normalization
        average = K.pool2d(squared, (self.n, self.n), strides=(1, 1),
                           padding="same", pool_mode="avg")
        denom = K.pow(self.k + self.alpha * average, self.beta)
        return x / denom

    def compute_output_shape(self, input_shape):
        return input_shape


# test the test harness
x = np.random.randn(10, 10)
layer = Dropout(0.5)
y = test_layer(layer, x)
assert(x.shape == y.shape)

x = np.random.randn(10, 10, 3)
layer = ZeroPadding2D(padding=(1, 1))
y = test_layer(layer, x)
assert(x.shape[0] + 2 == y.shape[0])
assert(x.shape[1] + 2 == y.shape[1])

x = np.random.randn(10, 10)
layer = Reshape((5, 20))
y = test_layer(layer, x)
assert(y.shape == (5, 20))

# test custom layer
x = np.random.randn(225, 225, 3)
layer = LocalResponseNormalization()
y = test_layer(layer, x)
assert(x.shape == y.shape)
