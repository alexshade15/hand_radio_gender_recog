from tensorflow.compat.v2.keras.models import Sequential
from tensorflow.compat.v2.keras.layers import *
from tensorflow.compat.v2.keras.optimizers import *
from tensorflow.compat.v2.keras.applications.vgg16 import VGG16
from tensorflow.compat.v2.keras.preprocessing.image import ImageDataGenerator
from tensorflow.compat.v2.keras import backend as K
from tensorflow.compat.v2.keras.callbacks import EarlyStopping
from tensorflow.compat.v2.keras.callbacks import TensorBoard

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import KFold
import numpy as np
import os
import random
import cv2
import sys
from datetime import datetime
import traceback
import kfold2 as kfolds


def csv_image_gen(dictLabs, listImages, imgPath, batch_size, lb, mode="train", aug=None):
    c = 0
    n1 = listImages  # List of training images
    random.shuffle(n1)
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
        c += batch_size
        if c + batch_size >= len(n1):
            c = 0
            random.shuffle(n1)
            if mode == "eval":
                break
        labels = lb.transform(np.array(labels))
        if aug is not None:
            (img, labels) = next(aug.flow(img, labels, batch_size=batch_size))
        #print(img, "\n\n\n", labels)
        yield img, labels


def load_model(unlock, weights, mode=0):
    vgg_conv = VGG16(include_top=False, weights='imagenet', input_shape=(512, 512, 3))

    #if unlock:
    #    for layer in vgg_conv.layers[:-4]:
    #        layer.trainable = False
    #else:
    for layer in vgg_conv.layers:
        layer.trainable = False
    vgg_conv.summary()

    model = Sequential()
    model.add(vgg_conv)

    if mode == 0:
        model.add(GlobalAveragePooling2D())
    else:
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    #for layer in model:
    #print(model.layers)
        #print(layer.input_shape, layer.output_shape, "\n\n")

    if weights is not None:
        model.load_weights(weights)

    if unlock:
        for layer in model.layers[0].layers[-4:]:
            layer.trainable = True

    return model


def main(epoch=10, bs=64, unlock=False, weights=None, optimizer=(SGD(), "SGD"), my_lr=0.001, my_momentum=0.9,
         my_nesterov=False, my_decay=0.0):
    try:
        # initialize the paths to our training and testing CSV files
        #trainCsvPath = "/data/train.csv"
        csvPath = "/data/new.csv"
        trainPath = '/data/r_r_handset/training/'
        # valPath = '/data/handset/validation1/'
        # testPath = '/data/handset/validation2/'

        # open the training CSV file, then initialize the unique set of class
        # labels in the dataset along with the testing labels
        f = open(csvPath, "r")
        f.readline()
        labels = set()
        csvLabs = {}
        for line in f:
            line_content = line.strip().split(",")
            label = line_content[1]
            csvLabs[line_content[0]] = line_content[1]
            labels.add(label)
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

        # kf = KFold(n_splits=5, shuffle=True)
        X = np.array(os.listdir(trainPath))

        # for train_index, val_index in kf.split(os.listdir(trainPath)):
        train_index = kfolds.training_fold0
        val_index = kfolds.validation_fold0
        print("################################", len(train_index), len(val_index))
        NUM_TRAIN_IMAGES = len(train_index)
        NUM_VAL_IMAGES = len(val_index)

        #print("\n\n", NUM_TRAIN_IMAGES, NUM_VAL_IMAGES)

        trainingImages = X[train_index]
        validationImages = X[val_index]

        # initialize both the trainvving and testing image generators
        trainGen = csv_image_gen(csvLabs, trainingImages, trainPath, bs, lb, mode="train", aug=None)
        valGen = csv_image_gen(csvLabs, validationImages, trainPath, bs, lb, mode="train", aug=None)
        # testGen = csv_image_gen(csvLabs, os.listdir(testImages), trainPath, bs, lb, mode="eval", aug=None)

        model = load_model(unlock, weights, 0)

        NUM_TEST_IMAGES = NUM_VAL_IMAGES
        opt = optimizer[1]
        my_opt = optimizer[0]
        model.compile(loss='binary_crossentropy', optimizer=my_opt, metrics=['accuracy'])

        tbCallBack = TensorBoard(log_dir="log_NEWFOLD_ADAM_tb_4_3", write_graph=True, write_images=True)
        # es = EarlyStopping(monitor='val_loss', verbose=1, patience=20)

        history = model.fit_generator(trainGen, epochs=epoch, verbose=1, callbacks=[tbCallBack],
                                      validation_data=valGen,
                                      validation_steps=(NUM_VAL_IMAGES // bs),
                                      steps_per_epoch=(NUM_TRAIN_IMAGES // bs))

        score = [0, 0] #model.evaluate_generator(valGen, NUM_TEST_IMAGES // bs)

        dateTimeObj = datetime.now()

        name_model = "_opt:" + str(opt) + "_ep:" + str(epoch) + "_bs:" + str(bs) + "_lr:" + str(
            my_lr) + "_mom:" + str(my_momentum) + "_nest:" + str(my_nesterov) + "_dec:" + str(
            my_decay) + "_unlock:" + str(unlock) + "_acc:" + str(score[1]) + "_loss:" + str(score[0])
        weights_name = 'models_kfold_vgg/fine_vgg16_' + name_model + "_date:" + str(dateTimeObj) + '.h5'
        model.save(weights_name)
        f = open("models_kfold_vgg/training_log" + name_model + "_date:" + str(dateTimeObj) + ".txt", "w+")
        f.write("train_acc = " + str(history.history['accuracy']) + "\n")
        f.write("valid_acc = " + str(history.history['val_accuracy']) + "\n")
        f.write("train_loss = " + str(history.history['loss']) + "\n")
        f.write("valid_loss = " + str(history.history['val_loss']) + "\n")
        f.write("Score\n")
        f.write("Loss test " + str(score[0]) + "\n")
        f.write("Acc test " + str(score[1]) + "\n\n\n")
        f.write("training_set = " + str(train_index) + "\n")
        f.write("test_set = " + str(val_index) + "\n")
        f.close()
        return weights_name
    except Exception:
        f = open("models_kfold_vgg/error_log" + name_model + "_date:" + str(dateTimeObj) + ".txt", "w+")
        f.write(traceback.format_exc())
        f.write(str(sys.exc_info()[2]))
        f.close()
        print(traceback.format_exc())
        print(sys.exc_info()[2])


if __name__ == "__main__":
    try:
        epoch = int(sys.argv[1])
    except:
        epoch = 50
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
        weights = "pesi_new_kfold_76-75.h5"
    print("epoch: %d, batch_size: %d, unlock: %s, weights: %s \n\n" % (epoch, batch_size, unlock, weights))

    optimizers = []
    lrs = [0.01, 0.1,
           0.00001, 0.0001, 0.001,
           0.0001, 0.001,
           0.01, 0.001, 0.0001,
           0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
           0.001, 0.001, 0.001, 0.001, 0.001, 0.001,
           0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001,
           0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
           0.01, 0.001, 0.0001,
           0.01, 0.001, 0.0001]
    moms = [None, None,
            None, None, None,
            None, None,
            .9, .9, .9,
            .0, .2, .4, .6, .8, .9,
            .0, .2, .4, .6, .8, .9,
            .0, .2, .4, .6, .8, .9,
            .0, .2, .4, .6, .8, .9,
            .9, .9, .9,
            .9, .9, .9]
    nesterovs = [None, None,
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
              5e-5, 1e-6, 1e-6]

    optimizers.append((Adam(lr=lrs[0]), "Adam"))
    optimizers.append((Adam(lr=lrs[1]), "Adam"))
    optimizers.append((Adam(lr=lrs[2]), "Adam"))
    optimizers.append((Adam(lr=lrs[3]), "Adam"))
    optimizers.append((Adam(lr=lrs[4]), "Adam"))
    optimizers.append((RMSprop(lr=lrs[5]), "RMSprop"))
    optimizers.append((RMSprop(lr=lrs[6]), "RMSprop"))

    optimizers.append((SGD(lr=lrs[7], momentum=moms[7], nesterov=nesterovs[7], decay=decays[7]), "SGD"))
    optimizers.append((SGD(lr=lrs[8], momentum=moms[8], nesterov=nesterovs[8], decay=decays[8]), "SGD"))
    optimizers.append((SGD(lr=lrs[9], momentum=moms[9], nesterov=nesterovs[9], decay=decays[9]), "SGD"))
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

    for i in [4,  3]:
        print("epochs: {}, bs: {}, unlock: {}, pesi: {}, opt: {}, lr: {}, mom: {}, nest: {}, dec: {}".format(epoch,batch_size, unlock, weights, optimizers[i], lrs[i], moms[i], nesterovs[i], decays[i]))
        main(epoch, batch_size, unlock, weights, optimizers[i], lrs[i], moms[i], nesterovs[i], decays[i])
        K.clear_session()
        print("Training succesfully")
