from numpy.random import seed

seed(42)
from tensorflow.compat.v1 import set_random_seed

set_random_seed(42)

import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model as lm
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard
# from tensorflow.compat.v2.keras.callbacks import EarlyStopping

from sklearn.preprocessing import LabelBinarizer
import numpy as np
import random
import cv2
import sys
import os
import traceback
from datetime import datetime


def csv_image_gen(dict_labs, list_images, imgPath, batch_size, lb, mode="train", aug=None):
    c = 0
    n1 = list_images  # List of training images
    if mode == "train":
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
            labels.append(dict_labs[number])
        c += batch_size
        if c + batch_size - 1 > len(n1):
            c = 0
            random.shuffle(n1)
            if mode == "eval":
                break
        labels = lb.transform(np.array(labels))
        if aug is not None:
            (img, labels) = next(aug.flow(img, labels, batch_size=batch_size))
        yield img, labels


def load_model(unlock, weights, mode=0, base_architecture="vgg16"):
    if base_architecture == "vgg16":
        base_net = VGG16(include_top=False, weights='imagenet', input_shape=(512, 512, 3))
    elif base_architecture == "resnet":
        base_net = ResNet50(include_top=False, weights='imagenet', input_shape=(512, 512, 3))

    for layer in base_net.layers[:]:
        layer.trainable = False
    base_net.summary()

    model = Sequential()
    model.add(base_net)

    if mode == 0:
        model.add(GlobalAveragePooling2D())
    else:
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    if weights is not None:
        model.lm(weights)

    if unlock >= 1:
        for layer in model.layers[0].layers[-(4 * unlock):]:
            layer.trainable = True
    for layer in model.layers[0].layers:
        print(layer.trainable)

    return model


def get_labels(csv_path):
    f = open(csv_path, "r")
    # removes the first line of the the csv, which is "id,male"
    f.readline()
    labels = set()
    csv_labs = {}
    for line in f:
        line_content = line.strip().split(",")
        label = line_content[1]
        csv_labs[line_content[0]] = line_content[1]
        labels.add(label)
    f.close()
    # create the label binarizer for one-hot encoding labels, then encode the testing labels
    lb = LabelBinarizer()
    lb.fit(list(labels))
    return lb, csv_labs


def do_training(epoch, batch_size, optimizer, my_lr, my_momentum, my_nesterov, my_decay, unlock, weights, csv_path,
                training_images, train_path, validation_images, val_path, test_images, test_path, log_name,
                base_architecture="vgg16"):
    lb, csv_labs = get_labels(csv_path)

    train_gen = csv_image_gen(csv_labs, training_images, train_path, batch_size, lb, mode="train", aug=None)
    val_gen = csv_image_gen(csv_labs, validation_images, val_path, batch_size, lb, mode="train", aug=None)
    test_gen = csv_image_gen(csv_labs, test_images, test_path, batch_size, lb, mode="eval", aug=None)

    num_train_images = len(training_images)
    num_val_images = len(validation_images)
    num_test_images = len(test_images)

    model = load_model(unlock, weights, 0, base_architecture=base_architecture)

    opt = optimizer[1]
    my_opt = optimizer[0]
    model.compile(loss='binary_crossentropy', optimizer=my_opt, metrics=['accuracy'])

    tb_call_back = TensorBoard(log_dir="log_" + log_name, write_graph=True, write_images=True)
    # es = EarlyStopping(monitor='val_loss', verbose=1, patience=20)

    history = model.fit_generator(train_gen, epochs=epoch, verbose=1, callbacks=[tb_call_back],
                                  validation_data=val_gen,
                                  validation_steps=(num_val_images // batch_size),
                                  steps_per_epoch=(num_train_images // batch_size))

    score = model.evaluate_generator(test_gen, num_test_images // batch_size)
    print("Score:", score)

    write_info(model, score, history, epoch, batch_size, unlock, opt, my_lr, my_momentum, my_nesterov, my_decay)


def write_info(model, score, history, epoch, bs, opt, my_lr, my_momentum=None, my_nesterov=None, my_decay=None,
               unlock=None):
    try:
        date_time_obj = datetime.now()
        name_model = "_opt:" + str(opt) + "_ep:" + str(epoch) + "_bs:" + str(bs) + "_lr:" + str(my_lr)
        if (my_momentum and my_nesterov and my_decay) is not None:
            name_model += "_mom:" + str(my_momentum) + "_nest:" + str(my_nesterov) + "_dec:" + str(my_decay)
        if unlock is not None:
            name_model += "_unlock:" + str(unlock)
        name_model += str(score[1]) + "_loss:" + str(score[0])

        weights_name = 'models_kfold_vgg/fine_vgg16' + name_model + "_date:" + str(date_time_obj) + '.h5'
        model.save(weights_name)
        f = open("models_kfold_vgg/training_log" + name_model + "_date:" + str(date_time_obj) + ".txt", "w+")
        f.write("train_acc = " + str(history.history['accuracy']) + "\n")
        f.write("valid_acc = " + str(history.history['val_accuracy']) + "\n")
        f.write("train_loss = " + str(history.history['loss']) + "\n")
        f.write("valid_loss = " + str(history.history['val_loss']) + "\n")
        f.write("Score\n")
        f.write("Loss test " + str(score[0]) + "\n")
        f.write("Acc test " + str(score[1]))
        f.close()
    except Exception:
        f = open("models_kfold_vgg/error_log" + name_model + "_date:" + str(date_time_obj) + ".txt", "w+")
        f.write(traceback.format_exc())
        f.write(str(sys.exc_info()[2]))
        f.close()
        print(traceback.format_exc())
        print(sys.exc_info()[2])


def generate_performance(predictions_test, test, csv_labs, samples):
    male = 0
    female = 0
    classified_as_male = {}
    classified_as_female = {}
    male_classified_as_female = []
    female_classified_as_male = []

    for index, elem in enumerate(predictions_test):
        key = test[index].split(".")[0]
        if elem > .5:
            classified_as_male[key] = csv_labs[key]
        else:
            classified_as_female[key] = csv_labs[key]

    for key in test:
        key = key.split(".")[0]
        if csv_labs[key] == "True":
            male += 1
        else:
            female += 1

    for key in classified_as_male:
        if classified_as_male[key] == "False":
            female_classified_as_male.append(key)

    for key in classified_as_female:
        if classified_as_female[key] == "True":
            male_classified_as_female.append(key)

    male_correct_classified = len(classified_as_male) - len(female_classified_as_male)
    female_correct_classified = len(classified_as_female) - len(male_classified_as_female)
    overall_accuracy = (male_correct_classified + female_correct_classified) / samples
    male_accuracy = male_correct_classified / male
    female_accuracy = female_correct_classified / female

    print("male", male, "female", female, "samples", samples)
    print("overall_accuracy:", overall_accuracy)
    print("male_accuracy:", male_accuracy)
    print("female_accuracy:", female_accuracy, "\n")

    print(male_accuracy, "|", len(female_classified_as_male) / samples)
    print(len(male_classified_as_female) / samples, "|", female_accuracy)

    print("\nmale_classified_as_female\n", male_classified_as_female)
    print("female_classified_as_male\n", female_classified_as_male)

def test(model_name):
    # evaluate the model: overall accuracy, accuracy on the single class
    # write the wrong classified
    # generate the confusion matrix

    csv_path = "/data/unified.csv"
    test1_path = '/data/handset/validation2/'
    test2_path = '/data/original_r2_handset/validation2/'

    # csv_path = "/Users/alex/Desktop/full.csv"
    # test1_path = "/Users/alex/Desktop/bone age/validation/boneage-validation-dataset-2"

    test1 = os.listdir(test1_path)
    samples = len(test1)
    test2 = os.listdir(test2_path)
    lb, csv_labs = get_labels(csv_path)
    test1_gen = csv_image_gen(csv_labs, test1, test1_path, 1, lb, mode="eval", aug=None)
    test2_gen = csv_image_gen(csv_labs, test2, test2_path, 1, lb, mode="eval", aug=None)

    model = lm(model_name)

    predictions_test1 = model.predict_generator(test1_gen, samples)
    predictions_test2 = model.predict_generator(test2_gen, len(test2))
    print("\n\n")
    print("TEST1 ON /data/handset/validation2/, original dataset\n")
    generate_performance(predictions_test1, test1, csv_labs, samples)
    print("\n\n")
    print("TEST2 ON /data/original_r2_handset/validation2/, cleaned dataset\n")
    generate_performance(predictions_test2, test2, csv_labs, samples)


# test("/Users/alex/Downloads/vgg16_open1_sgd29.h5")

aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

lrs = [0.1, 0.01,
       0.001, 0.0001, 0.0001,
       0.001, 0.0001,
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

optimizers = [(Adam(lr=lrs[0]), "Adam"),
              (Adam(lr=lrs[1]), "Adam"),
              (Adam(lr=lrs[2]), "Adam"),
              (Adam(lr=lrs[3]), "Adam"),
              (Adam(lr=lrs[4]), "Adam"),
              (RMSprop(lr=lrs[5]), "RMSprop"), (RMSprop(lr=lrs[6]), "RMSprop"),
              (SGD(lr=lrs[7], momentum=moms[7], nesterov=nesterovs[7], decay=decays[7]), "SGD"),
              (SGD(lr=lrs[8], momentum=moms[8], nesterov=nesterovs[8], decay=decays[8]), "SGD"),
              (SGD(lr=lrs[9], momentum=moms[9], nesterov=nesterovs[9], decay=decays[9]), "SGD"),

              # 0.01 - 2
              (SGD(lr=lrs[10], momentum=moms[10]), "SGD"),
              (SGD(lr=lrs[11], momentum=moms[11]), "SGD"),
              (SGD(lr=lrs[12], momentum=moms[12]), "SGD"),
              (SGD(lr=lrs[13], momentum=moms[13]), "SGD"),
              (SGD(lr=lrs[14], momentum=moms[14]), "SGD"),
              (SGD(lr=lrs[15], momentum=moms[15]), "SGD"),
              # 0.001 - 3
              (SGD(lr=lrs[16], momentum=moms[16]), "SGD"),
              (SGD(lr=lrs[17], momentum=moms[17]), "SGD"),
              (SGD(lr=lrs[18], momentum=moms[18]), "SGD"),
              (SGD(lr=lrs[19], momentum=moms[19]), "SGD"),
              (SGD(lr=lrs[20], momentum=moms[20]), "SGD"),
              (SGD(lr=lrs[21], momentum=moms[21]), "SGD"),
              # 0.0001 - 4
              (SGD(lr=lrs[22], momentum=moms[22]), "SGD"),
              (SGD(lr=lrs[23], momentum=moms[23]), "SGD"),
              (SGD(lr=lrs[24], momentum=moms[24]), "SGD"),
              (SGD(lr=lrs[25], momentum=moms[25]), "SGD"),
              (SGD(lr=lrs[26], momentum=moms[26]), "SGD"),
              (SGD(lr=lrs[27], momentum=moms[27]), "SGD"),
              # 0.00001 - 5
              (SGD(lr=lrs[28], momentum=moms[28]), "SGD"),
              (SGD(lr=lrs[29], momentum=moms[29]), "SGD"),
              (SGD(lr=lrs[30], momentum=moms[30]), "SGD"),
              (SGD(lr=lrs[31], momentum=moms[31]), "SGD"),
              (SGD(lr=lrs[32], momentum=moms[32]), "SGD"),
              (SGD(lr=lrs[33], momentum=moms[33]), "SGD"),

              (SGD(lr=lrs[34], momentum=moms[34], nesterov=nesterovs[34], decay=decays[34]), "SGD"),
              (SGD(lr=lrs[35], momentum=moms[35], nesterov=nesterovs[35], decay=decays[35]), "SGD"),
              (SGD(lr=lrs[36], momentum=moms[36], nesterov=nesterovs[36], decay=decays[36]), "SGD"),
              (SGD(lr=lrs[37], momentum=moms[37], nesterov=nesterovs[37], decay=decays[37]), "SGD"),
              (SGD(lr=lrs[38], momentum=moms[38], nesterov=nesterovs[38], decay=decays[38]), "SGD"),
              (SGD(lr=lrs[39], momentum=moms[39], nesterov=nesterovs[39], decay=decays[39]), "SGD")]
