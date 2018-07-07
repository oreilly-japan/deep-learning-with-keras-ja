import os
import sys
from urllib.request import urlretrieve
from keras.preprocessing.image import load_img, img_to_array
from keras import backend as K
from keras.applications import vgg16
import matplotlib.pyplot as plt
import numpy as np


def preprocess_image(img_path):
    img = load_img(img_path)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg16.preprocess_input(img)
    return img


def deprocess_image(x, gradient=False):
    img = x.copy()
    # Util function to convert a tensor into a valid image.
    img = img.reshape(x.shape[1], x.shape[2], x.shape[3])  # H, W, C
    if gradient:
        img = (img - img.mean()) * 255 / img.std()
    else:
        # Remove zero-center by mean pixel
        img[:, :, 0] += 103.939
        img[:, :, 1] += 116.779
        img[:, :, 2] += 123.68

    # 'BGR'->'RGB'
    img = img[:, :, ::-1]
    img = np.clip(img, 0, 255).astype("uint8")
    return img


def main(image_path, num_pool_layers=5, iter_count=3, learning_rate=10,
         show_rand=False):
    image = preprocess_image(image_path)
    model = vgg16.VGG16(weights="imagenet", include_top=False)
    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    fig = plt.figure(figsize=(17, 8))
    dream = model.input
    for i in range(num_pool_layers):
        _image = image.copy()
        rand_input = np.random.randint(
                        100, 150, size=_image.shape, dtype=np.uint8)
        layer_name = "block{:d}_pool".format(i + 1)
        layer_output = layer_dict[layer_name].output
        loss = K.mean(layer_output)
        grads = K.gradients(loss, dream)[0]
        grads /= K.maximum(K.mean(K.abs(grads)), 1e-5)  # normalize grad
        converter = K.function([dream], [loss, grads])

        grad_sum = None
        for j in range(iter_count):
            _loss_value, _grads_value = converter([_image])
            _image += _grads_value * learning_rate  # gradient "ascent"
            if show_rand:
                _, _grads_value = converter([rand_input])
            if grad_sum is None:
                grad_sum = _grads_value
            else:
                grad_sum += _grads_value
        grad_mean = grad_sum / iter_count

        ax = plt.subplot(2, num_pool_layers, i + 1)
        ax.imshow(deprocess_image(_image))
        ax.axis("off")
        ax.set_title("dream from {}".format(layer_name))

        ax = plt.subplot(2, num_pool_layers, num_pool_layers + i + 1)
        ax.imshow(deprocess_image(grad_mean, gradient=True))
        ax.axis("off")
        ax.set_title("{}'s gradient".format(layer_name))

    plt.tight_layout()
    dir_name, file_name = os.path.split(image_path)
    file_root, ext = os.path.splitext(file_name)
    plt.savefig(os.path.join(dir_name, file_root + "_deep_dream.png"))
    plt.show()


if __name__ == "__main__":
    # cat image url
    image_url = "http://farm2.static.flickr.com/1200/525304657_c59f741aac.jpg"
    data_path = os.path.join(os.path.dirname(__file__), "data/cat.jpg")
    urlretrieve(image_url, data_path)
    if len(sys.argv) > 1 and sys.argv[1] == "--rand":
        main(data_path, show_rand=True)
    else:
        main(data_path)
