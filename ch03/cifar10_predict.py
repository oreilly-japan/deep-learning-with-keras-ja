from pathlib import Path
import numpy as np
from PIL import Image
from keras.models import load_model


model_path = "logdir_cifar10_deep_with_aug/model_file.hdf5"
images_folder = "sample_images"

# load model
model = load_model(model_path)
image_shape = (32, 32, 3)


# load images
def crop_resize(image_path):
    image = Image.open(image_path)
    length = min(image.size)
    crop = image.crop((0, 0, length, length))
    resized = crop.resize(image_shape[:2])  # use width x height
    img = np.array(resized).astype("float32")
    img /= 255
    return img


folder = Path(images_folder)
image_paths = [str(f) for f in folder.glob("*.png")]
images = [crop_resize(p) for p in image_paths]
images = np.asarray(images)

predicted = model.predict_classes(images)

assert predicted[0] == 3, "image should be cat."
assert predicted[1] == 5, "image should be dog."

print("You can detect cat & dog!")
