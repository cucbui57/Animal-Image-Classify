
from tqdm import tqdm

import os
from keras.models import load_model
from skimage.io import imread
from scipy.spatial import distance
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Model
from keras import backend as K
from skimage import transform
import warnings

warnings.filterwarnings("ignore")

model = load_model('models/my_model.h5')


def get_animal_name(label):
    if label == 1:
        return "butterfly"
    if label == 2:
        return "Fish"
    if label == 3:
        return "spider"
    if label == 0:
        return "Fox_dog"
    if label == 4:
        return "cat"


def save_feature_to_file():
    for folder in tqdm(os.listdir("./dataset/"), desc="Saving"):
        for image in tqdm(os.listdir(os.path.join("./dataset/", folder, "images")),
                                   desc=folder):
            paths = os.path.join('./dataset/', folder, 'images', image)
            img = imread(paths)
            img = reshaped_image(img)
            a = []
            a.append(img)
            a = np.array(a)
            layer_name = 'activation_4'
            intermediate_layer_model = Model(inputs=model.input,
                                             outputs=model.get_layer(layer_name).output)
            intermediate_output = intermediate_layer_model.predict(a)
            #         print(intermediate_output[0])
            if not os.path.exists(os.path.join('./dataset/', folder, 'feature')):
                os.mkdir(os.path.join('./dataset/', folder, 'feature'))
            np.save(os.path.join('./dataset/', folder, 'feature', image[:-5]), intermediate_output[0])


def reshaped_image(image):
    return transform.resize(image, (64, 64, 3))


def best_match_image(src):
    src = reshaped_image(src)
    a = []
    a.append(src)
    a = np.array(a)
    layer_name = 'activation_4'
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(a)
    dst = src
    featsrc = intermediate_output[0]

    mindist = 10000000000000000
    label = np.argmax(model.predict(a), axis=1)
    animal_name = get_animal_name(label[0])
    for filename in os.listdir(os.path.join("./dataset/", animal_name, "feature")):
        featdst = np.load(os.path.join('./dataset/', animal_name, 'feature', filename))
        if mindist > distance.euclidean(featsrc, featdst):
            mindist = distance.euclidean(featsrc, featdst)
            dst = imread(os.path.join('./dataset/', animal_name, 'images', filename.replace(filename[-3:], "JPEG")))
    return dst


# save_feature_to_file()

# src = imread('./dataset/cat/images/n02124075_305.JPEG')
src = imread('/home/local/python/CSDLDPT/test/fish/n01443537_405.JPEG')
fig, ax = plt.subplots(1, 2)

ax.flat[0].imshow(src)
ax.flat[0].set_title("Original Image")
ax.flat[1].imshow(best_match_image(src))
ax.flat[1].set_title("Most Similar Image")
plt.show()

