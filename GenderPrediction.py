import cv2
import model
import sys
import os

from tensorflow.compat.v2.keras.models import Sequential
from tensorflow.compat.v2.keras.layers import *
from tensorflow.compat.v2.keras.optimizers import *
from tensorflow.compat.v2.keras.applications.vgg16 import VGG16

# from sklearn.preprocessing import LabelBinarizer


def model(architecture, unlock, weights):
    vgg_conv = VGG16(include_top=False, weights='imagenet', input_shape=(512, 512, 3))

    if unlock:
        for layer in vgg_conv.layers[:-4]:
            layer.trainable = False
    else:
        for layer in vgg_conv.layers:
            layer.trainable = False

    m = Sequential()
    m.add(vgg_conv)
    if architecture:
        m.add(Flatten())
        m.add(Dense(512, activation='relu'))
        m.add(Dropout(0.5))
    else:
        model.add(GlobalAveragePooling2D())
    m.add(Dense(1, activation='sigmoid'))
    m.summary()

    if weights is not None:
        m.load_weights(weights)
    return m


if __name__ == "__main__":
    architecture = sys.argv[1]
    unlock = sys.argv[2]
    weights = sys.argv[3]
    folder = sys.argv[4]
    index = int(sys.argv[5])

    if "validation" in folder:
        csvPath = "/data/val.csv"
        a = 1
    else:
        csvPath = "/data/train.csv"
        a = 2

    f = open(csvPath, "r")
    f.readline()
    labels = set()
    dictLabs = {}
    for line in f:
        line_content = line.strip().split(",")
        # label = line_content[a]
        dictLabs[line_content[0]] = line_content[a]
        # labels.add(label)

    # lb = LabelBinarizer()
    # lb.fit(list(labels))

    m = model(architecture, unlock, weights)
    n = os.listdir(folder)

    img = cv2.imread(folder + '/' + n[index])
    img = cv2.resize(img / 255., (512, 512))
    img = img.reshape(512, 512, 3)

    number, ext = n[index].split(".")
    print("Image:", n[index])
    print("Label:", dictLabs[number])

    print("Prediction:", m.predict(img, verbose=1))
    print("Ground Truth:", labs[index])
