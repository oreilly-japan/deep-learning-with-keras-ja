import os
import shutil
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

path = os.path.join(os.path.dirname(__file__), "augs")
if os.path.exists(path):
    shutil.rmtree(path)

os.mkdir(path)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

indices = np.arange(x_train.shape[0])
np.random.shuffle(indices)

samples = x_train[indices[:5], :]

datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=False)

g = datagen.flow(
    samples,
    np.arange(len(samples)), batch_size=1,
    save_to_dir=path, save_prefix="auged_"
    )

for i in range(18):
    g.next()
