import os
import keras
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.datasets import cifar10
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2


def network():
    base_model = InceptionV3(weights="imagenet", include_top=False)
    for layer in base_model.layers:
        layer.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu")(x)
    prediction = Dense(10, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=prediction)
    return model


class CIFAR10Dataset():

    def __init__(self):
        """
        Setting image size for inceptionv3
        Reference
            https://keras.io/applications/#inceptionv3
        """
        self.image_shape = (190, 190, 3)
        self.num_classes = 10

    def upscale(self, x, data_size):
        data_upscaled = np.zeros((data_size,
                                  self.image_shape[0],
                                  self.image_shape[1],
                                  self.image_shape[2]))
        for i, img in enumerate(x):
            large_img = cv2.resize(img, dsize=(self.image_shape[0],
                                               self.image_shape[1]),)
            data_upscaled[i] = large_img
        return data_upscaled

    def get_batch(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        x_train = self.upscale(x_train, x_train.shape[0])
        x_test = self.upscale(x_test, x_test.shape[0])

        x_train, x_test = [self.preprocess(d) for d in [x_train, x_test]]
        y_train, y_test = [self.preprocess(d, True) for d in [y_train, y_test]]

        return x_train, y_train, x_test, y_test

    def preprocess(self, data, label_data=False):
        if label_data:
            # convert class vectors to binary class matrices
            data = keras.utils.to_categorical(data, self.num_classes)
        else:
            data = data.astype("float32")
            data /= 255  # convert the value to 0~1 scale
            shape = (data.shape[0],) + self.image_shape  # add dataset length
            data = data.reshape(shape)

        return data


class Trainer():

    def __init__(self, model, loss, optimizer):
        self._target = model
        self._target.compile(
            loss=loss, optimizer=optimizer, metrics=["accuracy"]
            )
        self.verbose = 1
        logdir = "logdir_" + os.path.basename(__file__).replace('.py', '')
        self.log_dir = os.path.join(os.path.dirname(__file__), logdir)
        self.model_file_name = "model_file.hdf5"

    def train(self, x_train, y_train, batch_size, epochs, validation_split):
        if os.path.exists(self.log_dir):
            import shutil
            shutil.rmtree(self.log_dir)  # remove previous execution
        os.mkdir(self.log_dir)

        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (0~180)
            width_shift_range=0.1,  # randomly shift images horizontally
            height_shift_range=0.1,  # randomly shift images vertically
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        # compute quantities for normalization (mean, std etc)
        datagen.fit(x_train)

        # split for validation data
        indices = np.arange(x_train.shape[0])
        np.random.shuffle(indices)
        validation_size = int(x_train.shape[0] * validation_split)
        x_train, x_valid = \
            x_train[indices[:-validation_size], :], \
            x_train[indices[-validation_size:], :]
        y_train, y_valid = \
            y_train[indices[:-validation_size], :], \
            y_train[indices[-validation_size:], :]

        model_path = os.path.join(self.log_dir, self.model_file_name)
        self._target.fit_generator(
            datagen.flow(x_train, y_train, batch_size=batch_size),
            steps_per_epoch=x_train.shape[0] // batch_size,
            epochs=epochs,
            validation_data=(x_valid, y_valid),
            callbacks=[
                TensorBoard(log_dir=self.log_dir),
                ModelCheckpoint(model_path, save_best_only=True),
                EarlyStopping(),
            ],
            verbose=self.verbose,
            workers=4
        )


dataset = CIFAR10Dataset()

# make model
model = network()

# train the model
x_train, y_train, x_test, y_test = dataset.get_batch()
trainer = Trainer(model, loss="categorical_crossentropy", optimizer=RMSprop())
trainer.train(
    x_train, y_train, batch_size=26, epochs=8, validation_split=0.2
    )
model = load_model(os.path.join(trainer.log_dir, trainer.model_file_name))

for layer in model.layers[:249]:
    layer.trainable = False
for layer in model.layers[249:]:
    layer.trainable = True

trainer = Trainer(model, loss="categorical_crossentropy",
                  optimizer=SGD(lr=0.001, momentum=0.9))
trainer.train(
    x_train, y_train, batch_size=26, epochs=8, validation_split=0.2
)
model = load_model(os.path.join(trainer.log_dir, trainer.model_file_name))

# show result
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
