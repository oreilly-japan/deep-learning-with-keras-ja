from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
import keras.preprocessing.image as Image
import numpy as np


model = VGG16(weights="imagenet", include_top=True)

image_path = "sample_images_pretrain/steaming_train.png"
image = Image.load_img(image_path, target_size=(224, 224))  # imagenet size
x = Image.img_to_array(image)
x = np.expand_dims(x, axis=0)  # add batch size dim
x = preprocess_input(x)

result = model.predict(x)
result = decode_predictions(result, top=3)[0]
print(result[0][1])  # show description
