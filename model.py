import os
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.preprocessing.image import ImageDataGenerator
# from IPython.display import display
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from skimage import io, transform
import warnings

warnings.filterwarnings("ignore")


def to_one_hot(labels, dimension=5):
    results = np.zeros((len(labels), dimension))

    for i, label in enumerate(labels):
        results[i, label] = 1.

    return results


def cnn_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3),padding ='same', input_shape=(64, 64, 3)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), padding = 'same'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding = 'same'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5))
    model.add(Activation('softmax'))

    # model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()
    return model


def reshaped_image(image):
    return transform.resize(image, (64, 64, 3))


def load_images_from_folder():
    train_images = []
    train_labels = []
    for (i, folder) in enumerate(os.listdir("./dataset/")):
        for image in os.listdir(os.path.join("./dataset/", folder, "images")):
            path = os.path.join('./dataset/', folder, 'images', image)
            img = io.imread(path)
            train_images.append(reshaped_image(img))
            train_labels.append([i])
    return np.array(train_images), np.array(train_labels)


train_data, train_labels = load_images_from_folder()
train_labels = to_one_hot(train_labels)

train_data, test_data, train_labels, test_labels = train_test_split(train_data, train_labels, test_size=0.1,
                                                                    random_state=42)
print("Train data size: ", len(train_data))
print("Test data size: ", len(test_data))

cnn = cnn_model()

print("Train data shape: ", train_data.shape)
print("Test data shape: ", test_data.shape)

cnn.fit(train_data, train_labels, batch_size=8, epochs=20)
predicted_test_labels = np.argmax(cnn.predict(test_data), axis=1)
test_labels = np.argmax(test_labels, axis=1)

print("Actual test labels:", test_labels)
print("Predicted test labels:", predicted_test_labels)

print("Accuracy score:", accuracy_score(test_labels, predicted_test_labels))
cnn.save("./models/my_model.h10")
