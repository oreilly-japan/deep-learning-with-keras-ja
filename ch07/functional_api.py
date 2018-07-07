from keras.layers import Input, Dense, Activation
from keras.models import Model

x = Input(shape=(784,))

g = Dense(32)  # 1
s_2 = Activation("sigmoid")  # 2
f = Dense(10)  # 3
s_K = Activation("softmax")  # 4
y = s_K(f(s_2(g(x))))

model = Model(inputs=x, outputs=y)
model.compile(loss="categorical_crossentropy", optimizer="adam")