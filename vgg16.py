from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2
import numpy as np


def vgg_16(weights_path=None):
    m = Sequential()
    m.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))
    m.add(Convolution2D(64, 3, 3, activation='relu'))
    m.add(ZeroPadding2D((1, 1)))
    m.add(Convolution2D(64, 3, 3, activation='relu'))
    m.add(MaxPooling2D((2, 2), strides=(2, 2)))

    m.add(ZeroPadding2D((1, 1)))
    m.add(Convolution2D(128, 3, 3, activation='relu'))
    m.add(ZeroPadding2D((1, 1)))
    m.add(Convolution2D(128, 3, 3, activation='relu'))
    m.add(MaxPooling2D((2, 2), strides=(2, 2)))

    m.add(ZeroPadding2D((1, 1)))
    m.add(Convolution2D(256, 3, 3, activation='relu'))
    m.add(ZeroPadding2D((1, 1)))
    m.add(Convolution2D(256, 3, 3, activation='relu'))
    m.add(ZeroPadding2D((1, 1)))
    m.add(Convolution2D(256, 3, 3, activation='relu'))
    m.add(MaxPooling2D((2, 2), strides=(2, 2)))

    m.add(ZeroPadding2D((1, 1)))
    m.add(Convolution2D(512, 3, 3, activation='relu'))
    m.add(ZeroPadding2D((1, 1)))
    m.add(Convolution2D(512, 3, 3, activation='relu'))
    m.add(ZeroPadding2D((1, 1)))
    m.add(Convolution2D(512, 3, 3, activation='relu'))
    m.add(MaxPooling2D((2, 2), strides=(2, 2)))

    m.add(ZeroPadding2D((1, 1)))
    m.add(Convolution2D(512, 3, 3, activation='relu'))
    m.add(ZeroPadding2D((1, 1)))
    m.add(Convolution2D(512, 3, 3, activation='relu'))
    m.add(ZeroPadding2D((1, 1)))
    m.add(Convolution2D(512, 3, 3, activation='relu'))
    m.add(MaxPooling2D((2, 2), strides=(2, 2)))

    m.add(Flatten())
    m.add(Dense(4096, activation='relu'))
    m.add(Dropout(0.5))
    m.add(Dense(4096, activation='relu'))
    m.add(Dropout(0.5))
    # m.add(Dense(1000, activation='softmax'))
    m.add(GlobalAveragePooling2D())
    m.add(Dense(2, activation='softmax'))

    if weights_path:
        m.load_weights(weights_path)

    return m


if __name__ == "__main__":
    im = cv2.resize(cv2.imread('cat.jpg'), (224, 224)).astype(np.float32)
    im[:, :, 0] -= 103.939
    im[:, :, 1] -= 116.779
    im[:, :, 2] -= 123.68
    im = im.transpose((2, 0, 1))
    im = np.expand_dims(im, axis=0)

    # Test pretrained model
    m = vgg_16('vgg16_weights.h5')
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    m.compile(optimizer=sgd, loss='categorical_crossentropy')
    out = m.predict(im)
    print(np.argmax(out))
