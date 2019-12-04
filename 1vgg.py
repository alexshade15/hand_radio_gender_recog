from tensorflow.compat.v2.keras.models import Sequential
from tensorflow.compat.v2.keras.layers import *
from tensorflow.compat.v2.keras.optimizers import *
from tensorflow.compat.v2.keras.applications.vgg16 import VGG16
from tensorflow.compat.v2.keras.preprocessing.image import ImageDataGenerator
from tensorflow.compat.v2.keras import backend as K
from tensorflow.compat.v2.keras.callbacks import EarlyStopping

from sklearn.preprocessing import LabelBinarizer
import numpy as np
import os
import random
import cv2
import traceback
import sys
from datetime import datetime
import traceback
from datetime import date


class Unbuffered(object):
    def __init__(self, filename):
        self.stdout = sys.stdout
        self.log = open(filename, "a")
        self.log.write("\n\n\n--------   " + datetime.now().strftime("%d/%m/%Y-%H:%M:%S").strip())

    def write(self, data):
        self.stdout.write(data)
        self.stdout.flush()
        self.log.write(data)

    def __enter__(self):
        sys.stdout = self

    def __exit__(self, exc_type, exc_value, tb):
        sys.stdout = self.stdout
        if exc_type is not None:
            self.log.write(traceback.format_exc())
        self.log.close()

    def flush(self):
        self.log.flush()
        self.stdout.flush()


def csv_image_generator(dictLabs, imgPath, batch_size, lb, mode="train", aug=None):
    c = 0
    n1 = os.listdir(imgPath)  # List of training images
    random.shuffle(n1)  # n2 = os.listdir(mask_folder)  # List of training images
    while True:
        labels = []
        img = np.zeros((batch_size, 512, 512, 3)).astype('float')
        for i in range(c, c + batch_size):
            train_img = cv2.imread(imgPath + '/' + n1[i])
            train_img = cv2.resize(train_img / 255., (512, 512))  # Read an image from folder and resize
            train_img = train_img.reshape(512, 512, 3)
            number, ext = n1[i].split(".")
            img[i - c] = train_img  # add to array - img[0], img[1], and so on.
            labels.append(dictLabs[number])
            # print(n1[i] + "  --  " + dictLabs[number])
        c += batch_size
        if c + batch_size >= len(os.listdir(imgPath)):
            c = 0
            random.shuffle(n1)
            if mode == "eval":
                break
        labels = lb.transform(np.array(labels))
        if aug is not None:
            (img, labels) = next(aug.flow(img, labels, batch_size=batch_size))
        yield img, labels

    # # open the CSV file for reading
    # f = open(csvPath, "r")
    # # loop indefinitely
    # while True:
    #     # initialize our batches of images and labels
    #     images = []
    #     labels = []
    #     # keep looping until we reach our batch size
    #     while len(images) < batch_size:
    #         # attempt to read the next line of the CSV file
    #         line = f.readline()
    #
    #         # check to see if the line is empty, indicating we have
    #         # reached the end of the file
    #         if line == "":
    #             # reset the file pointer to the beginning of the file
    #             # and re-read the line
    #             f.seek(0)
    #             line = f.readline()
    #
    #             # if we are evaluating we should now break from our
    #             # loop to ensure we don't continue to fill up the
    #             # batch from samples at the beginning of the file
    #             if mode == "eval":
    #                 break
    #
    #         # extract the label and construct the image
    #         line = line.strip().split(",")
    #         label = line[0]
    #         image = np.array([int(x) for x in line[1:]], dtype="uint8")
    #         image = image.reshape((64, 64, 3))
    #
    #         # update our corresponding batches lists
    #         images.append(image)
    #         labels.append(label)
    #         # one-hot encode the labels
    #         labels = lb.transform(np.array(labels))
    #
    #         # if the data augmentation object is not None, apply it
    #         if aug is not None:
    #             (images, labels) = next(aug.fl

def main(epoch=10, bs=64, unlock=False, weights=None, optimizer=(SGD(), "SGD"), lr=0.001, mom=0.9, nesterov=False, decay=0.0):
    try:
        #sys.stdout=Unbuffered("console_log_vgg16" + str(date.today().strftime("%d:%m")).strip() + ".txt")
        # initialize the paths to our training and testing CSV files
        trainCsvPath = "/data/train.csv"
        valCsvPath = "/data/val.csv"
        trainPath = '/data/handset/training/'
        valPath = '/data/handset/validation1/'
        testPath = '/data/handset/validation2/'

        # initialize the number of epochs to train for and batch size
        NUM_EPOCHS = epoch
        BATCH_SIZE = bs

        # initialize the total number of training and testing image
        NUM_TRAIN_IMAGES = len(os.listdir(trainPath))
        NUM_VAL_IMAGES = len(os.listdir(valPath))
        NUM_TEST_IMAGES = len(os.listdir(testPath))

        # open the training CSV file, then initialize the unique set of class
        # labels in the dataset along with the testing labels
        f = open(trainCsvPath, "r")
        f.readline()
        labels = set()
        trainLabs = {}
        valLabs = {}
        for line in f:
            # extract the class label, update the labels list, and increment
            # the total number of training images
            line_content = line.strip().split(",")
            label = line_content[2]
            trainLabs[line_content[0]] = line_content[2]
            labels.add(label)
        f.close()
        f = open(valCsvPath, "r")
        f.readline()
        for line in f:
            # extract the class label, update the test labels list, and
            # increment the total number of testing images
            line_content = line.strip().split(",")
            label = line_content[1]
            valLabs[line_content[0]] = line_content[1]
        f.close()

        # create the label binarizer for one-hot encoding labels, then encode the testing labels
        lb = LabelBinarizer()
        lb.fit(list(labels))

        # construct the training image generator for data augmentation
        aug = ImageDataGenerator(
            rotation_range=20,
            zoom_range=0.15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            horizontal_flip=True,
            fill_mode="nearest")

        # initialize both the training and testing image generators
        trainGen = csv_image_generator(trainLabs, trainPath, BATCH_SIZE, lb, mode="train", aug=None)
        valGen = csv_image_generator(valLabs, valPath, BATCH_SIZE, lb, mode="train", aug=None)
        testGen = csv_image_generator(valLabs, testPath, BATCH_SIZE, lb, mode="train", aug=None)

        # initialize our Keras model and compile it
        vgg_conv = VGG16(include_top=False, weights='imagenet', input_shape=(512, 512, 3))

        if unlock:
            for layer in vgg_conv.layers[:-4]:
                layer.trainable = False
        else:
            for layer in vgg_conv.layers:
                layer.trainable = False

        model = Sequential()
        model.add(vgg_conv)

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        #model.add(GlobalAveragePooling2D())
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        if weights is not None:
            model.load_weights(weights)
        my_lr = lr
        my_decay = decay
        my_momentum = mom
        my_nesterov = nesterov
        opt = optimizer[1]
        my_opt = optimizer[0]
        model.compile(loss='binary_crossentropy', optimizer=my_opt, metrics=['accuracy'])

        es = EarlyStopping(monitor='val_loss', verbose=1, patience=25)

        history = model.fit_generator(trainGen, epochs=NUM_EPOCHS, verbose=1, callbacks=[es],
                                      steps_per_epoch=(NUM_TRAIN_IMAGES // BATCH_SIZE),
                                      validation_data=valGen,
                                      validation_steps=(NUM_VAL_IMAGES // BATCH_SIZE))

        score = model.evaluate_generator(testGen, NUM_TEST_IMAGES // BATCH_SIZE)

        dateTimeObj = datetime.now()

        if unlock is None:
            isUnlocked = False
        else:
            isUnlocked = True

        name_model = "_opt:" + str(opt) + "_ep:" + str(epoch) + "_bs:" + str(bs) + "_lr:" + str(my_lr) + "_mom:" + str(my_momentum) + "_nest:" + str(my_nesterov) + "_dec:" + str(my_decay) + "_unlock:" + str(isUnlocked) + "_acc:" + str(score[1]) + "_loss:" + str(score[0])
        weights_name = 'models_vgg/fine_vgg16' + name_model + "_date:" + str(dateTimeObj) + '.h5'
        model.save(weights_name)
        try:
            f = open("models_vgg/training_log" + name_model + "_date:" + str(dateTimeObj) + ".txt", "w+")
            f.write("train_acc = " + str(history.history['accuracy']) + "\n")
            f.write("valid_acc = " + str(history.history['val_accuracy']) + "\n")
            f.write("train_loss = " + str(history.history['loss']) + "\n")
            f.write("valid_loss = " + str(history.history['val_loss']) + "\n")
            f.write("Score\n")
            f.write("Loss test " + str(score[0]) + "\n")
            f.write("Acc test " + str(score[1]) + "\n")
            f.close()
        except Exception:
            print("Exception on: fine_vgg16" + name_model + "_date:" + str(dateTimeObj))
            print(traceback.format_exc())
            print(sys.exc_info()[2])
        return weights_name
    except Exception:
        f = open("models_vgg/error_log" + name_model + "_date:" + str(dateTimeObj) + ".txt", "w+")
        f.write(traceback.format_exc())
        f.write(str(sys.exc_info()[2]))
        f.close()
        print(traceback.format_exc())
        print(sys.exc_info()[2])


def test_generator(use_aug=True, bs=4):
    trainCsvPath = "/data/train.csv"
    valCsvPath = "/data/val.csv"
    trainPath = '/data/handset/train/'
    BATCH_SIZE = bs

    f = open(trainCsvPath, "r")
    f.readline()
    labels = set()
    trainLabs = {}
    valLabs = {}

    for line in f:
        line_content = line.strip().split(",")
        label = line_content[2]
        trainLabs[line_content[0]] = line_content[2]
        labels.add(label)
    f.close()

    f = open(valCsvPath, "r")
    f.readline()
    for line in f:
        line_content = line.strip().split(",")
        label = line_content[1]
        valLabs[line_content[0]] = line_content[1]
    f.close()

    lb = LabelBinarizer()
    lb.fit(list(labels))

    aug = None
    if use_aug:
        aug = ImageDataGenerator(
            rotation_range=20,
            zoom_range=0.15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            horizontal_flip=True,
            fill_mode="nearest")

    return csv_image_generator(trainLabs, trainPath, BATCH_SIZE, lb, mode="train", aug=aug)

if __name__ == "__main__":
    try:
        epoch = int(sys.argv[1])
    except:
        epoch = 100
    try:
        batch_size = int(sys.argv[2])
    except:
        batch_size = 64
    try:
        unlock = sys.argv[3]
    except:
        unlock = True
    try:
        weights = sys.argv[4]
    except:
        weights = None
    print("epoch: %d, batch_size: %d, unlock: %s, weights: %s \n\n" % (epoch, batch_size, unlock, weights))

    optimizers = []
    lrs = [0.0001, 0.0001,
           0.00001, 0.0001, 0.001,
           0.0001, 0.001,
           0.01, 0.001, 0.0001,
           0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
           0.001, 0.001, 0.001, 0.001, 0.001, 0.001,
           0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001,
           0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
           0.01, 0.001, 0.0001,
           0.01, 0.001, 0.0001]
    moms = [.9, .9,
            None, None, None,
            None, None,
            .9, .9, .9,
            .0, .2, .4, .6, .8, .9,
            .0, .2, .4, .6, .8, .9,
            .0, .2, .4, .6, .8, .9,
            .0, .2, .4, .6, .8, .9,
            .9, .9, .9,
            .9, .9, .9]
    nesterovs = [False, True,
                 None, None, None,
                 None, None,
                 True, True, True,
                 False, False, False, False, False, False,
                 False, False, False, False, False, False,
                 False, False, False, False, False, False,
                 False, False, False, False, False, False,
                 True, True, True,
                 True, True, True]
    decays = [None, None,
              None, None, None,
              None, None,
              1e-6, 1e-6, 1e-6,
              None, None, None, None, None, None,
              None, None, None, None, None, None,
              None, None, None, None, None, None,
              None, None, None, None, None, None,
              1e-6, 1e-6, 1e-6,
              1e-6, 1e-6, 1e-6]
    optimizers.append((SGD(lr=lrs[0], momentum=moms[0]), "SGD"))
    optimizers.append((SGD(lr=lrs[1], momentum=moms[1], nesterov=nesterovs[1]), "SGD"))
    optimizers.append((Adam(lr=lrs[2]), "Adam"))
    optimizers.append((Adam(lr=lrs[3]), "Adam"))
    optimizers.append((Adam(lr=lrs[4]), "Adam"))
    optimizers.append((RMSprop(lr=lrs[5]), "RMSprop"))
    optimizers.append((RMSprop(lr=lrs[6]), "RMSprop"))

    optimizers.append((SGD(lr=lrs[7], momentum=moms, nesterov=nesterovs[7], decay=decays[7]), "SGD"))
    optimizers.append((SGD(lr=lrs[8], momentum=moms, nesterov=nesterovs[8], decay=decays[8]), "SGD"))
    optimizers.append((SGD(lr=lrs[9], momentum=moms, nesterov=nesterovs[9], decay=decays[9]), "SGD"))
    optimizers.append((SGD(lr=lrs[10], momentum=moms[10]), "SGD"))
    optimizers.append((SGD(lr=lrs[11], momentum=moms[11]), "SGD"))
    optimizers.append((SGD(lr=lrs[12], momentum=moms[12]), "SGD"))
    optimizers.append((SGD(lr=lrs[13], momentum=moms[13]), "SGD"))
    optimizers.append((SGD(lr=lrs[14], momentum=moms[14]), "SGD"))
    optimizers.append((SGD(lr=lrs[15], momentum=moms[15]), "SGD"))

    optimizers.append((SGD(lr=lrs[16], momentum=moms[16]), "SGD"))
    optimizers.append((SGD(lr=lrs[17], momentum=moms[17]), "SGD"))
    optimizers.append((SGD(lr=lrs[18], momentum=moms[18]), "SGD"))
    optimizers.append((SGD(lr=lrs[19], momentum=moms[19]), "SGD"))
    optimizers.append((SGD(lr=lrs[20], momentum=moms[20]), "SGD"))
    optimizers.append((SGD(lr=lrs[21], momentum=moms[21]), "SGD"))

    optimizers.append((SGD(lr=lrs[22], momentum=moms[22]), "SGD"))
    optimizers.append((SGD(lr=lrs[23], momentum=moms[23]), "SGD"))
    optimizers.append((SGD(lr=lrs[24], momentum=moms[24]), "SGD"))
    optimizers.append((SGD(lr=lrs[25], momentum=moms[25]), "SGD"))
    optimizers.append((SGD(lr=lrs[26], momentum=moms[26]), "SGD"))
    optimizers.append((SGD(lr=lrs[27], momentum=moms[27]), "SGD"))

    optimizers.append((SGD(lr=lrs[28], momentum=moms[28]), "SGD"))
    optimizers.append((SGD(lr=lrs[29], momentum=moms[29]), "SGD"))
    optimizers.append((SGD(lr=lrs[30], momentum=moms[30]), "SGD"))
    optimizers.append((SGD(lr=lrs[31], momentum=moms[31]), "SGD"))
    optimizers.append((SGD(lr=lrs[32], momentum=moms[32]), "SGD"))
    optimizers.append((SGD(lr=lrs[33], momentum=moms[33]), "SGD"))

    optimizers.append((SGD(lr=lrs[34], momentum=moms[34], nesterov=nesterovs[34], decay=decays[34]), "SGD"))
    optimizers.append((SGD(lr=lrs[35], momentum=moms[35], nesterov=nesterovs[35], decay=decays[35]), "SGD"))
    optimizers.append((SGD(lr=lrs[36], momentum=moms[36], nesterov=nesterovs[36], decay=decays[36]), "SGD"))

    optimizers.append((SGD(lr=lrs[37], momentum=moms[37], nesterov=nesterovs[37], decay=decays[37]), "SGD"))
    optimizers.append((SGD(lr=lrs[38], momentum=moms[38], nesterov=nesterovs[38], decay=decays[38]), "SGD"))
    optimizers.append((SGD(lr=lrs[39], momentum=moms[39], nesterov=nesterovs[39], decay=decays[39]), "SGD"))

    k = [1, 2]
    for i in k: #len(lrs)):
        main(epoch, batch_size, unlock, weights, optimizers[i], lrs[i], moms[i], nesterovs[i], decays[i])
        K.clear_session()
    print("Training succesfully")
