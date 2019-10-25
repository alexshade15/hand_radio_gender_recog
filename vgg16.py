from tensorflow.compat.v2.keras.models import Sequential
from tensorflow.compat.v2.keras.layers import *
from tensorflow.compat.v2.keras.optimizers import *
from tensorflow.compat.v2.keras.applications.vgg16 import VGG16
from tensorflow.compat.v2.keras.preprocessing.image import ImageDataGenerator

# from keras.models import Sequential
# from keras.layers import *
# from keras.optimizers import *
# from keras.applications.vgg16 import VGG16
# from keras.preprocessing.image import ImageDataGenerator

from sklearn.preprocessing import LabelBinarizer
import numpy as np
import os
import random
import cv2
import traceback
import sys
from datetime import datetime


def csv_image_generator(dictLabs, imgPath, batch_size, lb, mode="train", aug=None):
    c = 0
    n1 = os.listdir(imgPath)  # List of training images
    random.shuffle(n1)  # n2 = os.listdir(mask_folder)  # List of training images
    while True:
        labels = []
        try:
            del img
        except:
            img = None
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
    #             (images, labels) = next(aug.flow(np.array(images), labels, batch_size=batch_size))
    #         yield (np.array(images), labels)


def main(epoch=10, bs=64, unlock=False, weights=None):
    try:
        # initialize the paths to our training and testing CSV files
        trainCsvPath = "/data/train.csv"
        valCsvPath = "/data/val.csv"
        trainPath = '/data/handset/train/'
        valPath = '/data/handset/val/'
        testPath = '/data/handset/test/'

        # initialize the number of epochs to train for and batch size
        NUM_EPOCHS = epoch
        BATCH_SIZE = bs
        image_size = 512

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
        trainGen = csv_image_generator(trainLabs, trainPath, BATCH_SIZE, lb, mode="train", aug=aug)
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
        model.add(GlobalAveragePooling2D())
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        if weights is not None:
            model.load_weights(weights)

        mySgd = SGD(lr=1e-3, decay=5e-5, momentum=0.9, nesterov=True)
        model.compile(loss='binary_crossentropy', optimizer=mySgd, metrics=['accuracy'])

        history = model.fit_generator(trainGen, epochs=NUM_EPOCHS, verbose=1,
                                      steps_per_epoch=(NUM_TRAIN_IMAGES // BATCH_SIZE),
                                      validation_data=valGen,
                                      validation_steps=(NUM_VAL_IMAGES // BATCH_SIZE))

        score = model.evaluate_generator(testGen, NUM_TEST_IMAGES // BATCH_SIZE)

        dateTimeObj = datetime.now()

        f = open("training_log" + str(dateTimeObj) + ".txt", "w+")
        f.write("history - accuracy:\n")
        f.write(str(history.history['accuracy']))
        f.write("\n\nscores:\n")
        f.write(str(score) + "\n")
        f.close()
        # Save the model
        weights_name = 'fine_vgg16_ep-' + epoch + '_bs-' + bs + '_unlock-' + unlock + str(dateTimeObj) + '.h5'
        model.save(weights_name)
        
        return weights_name
    except Exception:
        f = open("error_log" + str(dateTimeObj) + ".txt", "w+")
        f.write(traceback.format_exc())
        f.write(str(sys.exc_info()[2]))
        f.close()
        print(traceback.format_exc())
        print(sys.exc_info()[2])


def multiple_train(epoch=10, bs=64, unlock=False, weights=None):
    epoch_done = 0
    step = 3
    while epoch_done < epoch:
        if (epoch - epoch_done) >= step:
            weights = main(step, bs, unlock, weights)
        else:
            weights = main(epoch - epoch_done, bs, unlock, weights)
        epoch_done += step
        print("Epoch " + epoch_done + "/" + epoch)
