import tensorflow
from tensorflow.compat.v2.keras.layers import *
from tensorflow.compat.v2.keras.models import Model
from tensorflow.compat.v2.keras.optimizers import *
from tensorflow.compat.v2.keras.preprocessing import image
from tensorflow.compat.v2.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.compat.v2.keras.preprocessing.image import ImageDataGenerator
import os
import cv2
import random
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
import traceback
import sys
from datetime import datetime
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

def data_gen(img_folder, mask_folder, batch_size, aug=None):
    c = 0
    n1 = os.listdir(img_folder)  # List of training images
    random.shuffle(n1)  # n2 = os.listdir(mask_folder)  # List of training images
    while True:
        img = np.zeros((batch_size, 512, 512, 1)).astype('float')
        mask = np.zeros((batch_size, 512, 512, 1)).astype('float')
        for i in range(c, c + batch_size):  # initially from 0 to 16, c = 0.
            train_img = cv2.imread(img_folder + '/' + n1[i], cv2.IMREAD_GRAYSCALE) / 255.
            train_img = cv2.resize(train_img, (512, 512))  # Read an image from folder and resize
            train_img = train_img.reshape(512, 512, 1)
            name, ext = n1[i].split(".")
            _, number = name.split("s")
            img[i - c] = train_img  # add to array - img[0], img[1], and so on.
            train_mask = cv2.imread(mask_folder + '/masks' + number + "." + ext, cv2.IMREAD_GRAYSCALE) / 255.
            train_mask = cv2.resize(train_mask, (512, 512))
            train_mask = train_mask.reshape(512, 512, 1)
            mask[i - c] = train_mask
        c += batch_size
        if c + batch_size >= len(os.listdir(img_folder)):
            c = 0
            random.shuffle(n1)
        if aug is not None:
            (img, mask) = next(aug.flow(img, mask, batch_size=batch_size))
        yield img, mask


# tensorflow.debugging.set_log_device_placement(True)
def unet(pretrained_weights=None, input_size=(512, 512, 1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
    m = Model(inputs=inputs, outputs=conv10)
    # m.compile(optimizer=SGD(lr=0.015), loss='binary_crossentropy', metrics=['accuracy'])
    return m


def loadImages(path):
    data = []
    for img_name in os.listdir(path):
        img = image.load_img(path + img_name, target_size=(512, 512, 1))
        img = image.img_to_array(img)
        img = img / 255
        data.append(img)
    return np.array(data)


def gridSearch(batch_size=4):
    train_frame_path = '/data/segmentation/train_frames/'
    train_mask_path = '/data/segmentation/train_masks/'
    test_frame_path = '/data/segmentation/test_frames/'
    test_mask_path = '/data/segmentation/test_masks/'

    train_X = loadImages(train_frame_path)
    train_Y = loadImages(train_mask_path)
    test_X = loadImages(test_frame_path)
    test_Y = loadImages(test_mask_path)

    m = unet()
    model = KerasClassifier(build_fn=m, epochs=25, batch_size=batch_size, verbose=0)

    optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    # learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
    # momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
    # param_grid = dict(learn_rate=learn_rate, momentum=momentum)
    param_grid = dict(optimizer=optimizer)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=3)
    grid_result = grid.fit(train_X, train_Y)

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


def myGrid(epoch = 50, bs=4):
    # optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    # [[0.22073517739772797, 0.9589321], [3.4372099876403808, 0.7771452], [3.537072992324829, 0.77067107],
    #  [0.4104248046875, 0.7677431], [3.5604740619659423, 0.76915395], [0.5666816473007202, 0.77221453],
    #  [3.537072849273682, 0.77067107]]

    sys.stdout=Unbuffered("console_log" + str(date.today().strftime("%d:%m")).strip() + ".txt")

    learn_rate = [0.001] #, 0.01, 0.1, 0.2, 0.3]
    #learn_rate = [0.01]
    momentum = [0.8, 0.9] #[0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
    histories = []
    scores = []
    models = []

    train_datagen = ImageDataGenerator(
        #rescale=1. / 255,
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest")
    val_datagen = ImageDataGenerator()
    test_datagen = ImageDataGenerator()

    train_frame_path = '/data/segmentation2/train_frames/train/'
    train_mask_path = '/data/segmentation2/train_masks/train/'
    val_frame_path = '/data/segmentation2/val_frames/val/'
    val_mask_path = '/data/segmentation2/val_masks/val/'
    test_frame_path = '/data/segmentation2/test_frames/test/'
    test_mask_path = '/data/segmentation2/test_masks/test/'

    NO_OF_EPOCHS = epoch
    BATCH_SIZE = bs
    max_score = 0

    #train_image_gen = train_datagen.flow_from_directory(train_frame_path, batch_size=BATCH_SIZE)
    #train_mask_gen = train_datagen.flow_from_directory(train_mask_path, batch_size=BATCH_SIZE)
    #val_image_gen = val_datagen.flow_from_directory(val_frame_path, batch_size=BATCH_SIZE)
    #val_mask_gen = val_datagen.flow_from_directory(val_mask_path, batch_size=BATCH_SIZE)
    #test_image_gen = test_datagen.flow_from_directory(test_frame_path, batch_size=BATCH_SIZE)
    #test_mask_gen = test_datagen.flow_from_directory(test_mask_path, batch_size=BATCH_SIZE)
    try:
        for lr in learn_rate:
            for mom in momentum:
                m = unet()
                m.compile(optimizer=SGD(learning_rate=lr, momentum=mom), loss='binary_crossentropy', metrics=['accuracy'])

                train_gen = data_gen(train_frame_path, train_mask_path, batch_size=BATCH_SIZE)
                val_gen = data_gen(val_frame_path, val_mask_path, batch_size=BATCH_SIZE)
                test_gen = data_gen(test_frame_path, test_mask_path, batch_size=BATCH_SIZE)
                NO_OF_TRAINING_IMAGES = len(os.listdir(train_frame_path))
                NO_OF_VAL_IMAGES = len(os.listdir(val_frame_path))
                NO_OF_TEST_IMAGES = len(os.listdir(test_frame_path))
                #print("NO_OF_TRAINING_IMAGES: ", NO_OF_TRAINING_IMAGES)
                #print("NO_OF_VAL_IMAGES: ", NO_OF_VAL_IMAGES)
                #print("NO_OF_TEST_IMAGES: ", NO_OF_TEST_IMAGES, "\n\n")

                history = m.fit_generator(train_gen, epochs=NO_OF_EPOCHS,
                                          steps_per_epoch=(NO_OF_TRAINING_IMAGES // BATCH_SIZE),
                                          validation_data=val_gen, validation_steps=(NO_OF_VAL_IMAGES // BATCH_SIZE))
                score = m.evaluate_generator(test_gen, NO_OF_TEST_IMAGES // BATCH_SIZE)
                histories.append(history)
                scores.append(score)
                print("\n\nScore: " + str(score))
                print("train acc " + str(history.history['accuracy']))
                print("valid acc " + str(history.history['val_accuracy']))
                print("learningRate: ", lr, "\nmomentum: ", mom)
                #print("\nNO_OF_TRAINING_IMAGES: ", NO_OF_TRAINING_IMAGES)
                #print("NO_OF_VAL_IMAGES: ", NO_OF_VAL_IMAGES)
                #print("NO_OF_TEST_IMAGES: ", NO_OF_TEST_IMAGES, "\n\n")
                m.save("./models/seg" + "_lr_" + str(lr) + "_mom_" + str(mom) + "__"  + str(score[1]) + "_" + str(score[0]) + ".h5", "w+")
                try:
                    f = open("./models/seg" + "_lr_" + str(lr) + "_mom_" + str(mom) + "__" + str(score[1]) + "_" + str(score[0]) + ".txt", "w+")
                    f.write("HH-ACC\n")
                    f.write("train acc " + str(history.history['accuracy']) + "\n")
                    f.write("valid acc " + str(history.history['val_accuracy']) + "\n")
                    f.write("HH-LOSS\n")
                    f.write("train loss " + str(history.history['loss']) + "\n")
                    f.write("valid loss " + str(history.history['val_loss']) + "\n")
                    f.write("Score\n")
                    f.write("Loss test " + str(score[0]) + "\n")
                    f.write("Acc test " + str(score[1]) + "\n")
                    f.close()
                except Exception:
                    print("Exception on: ", "_lr: " + str(lr) + "_mom: " + str(mom))
                    print(traceback.format_exc())
                    print(sys.exc_info()[2])
    except Exception:
        f = open("error_log.txt", "w+")
        f.write(traceback.format_exc())
        f.write(str(sys.exc_info()[2]))
        f.close()
        print(traceback.format_exc())
        print(sys.exc_info()[2])



if __name__ == "__main__":
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=30,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")
    val_datagen = ImageDataGenerator()
    test_datagen = ImageDataGenerator()
    BATCH_SIZE = 4

    train_frame_path = '/data/segmentation2/train_frames/'
    train_mask_path = '/data/segmentation2/train_masks/'
    val_frame_path = '/data/segmentation2/val_frames/'
    val_mask_path = '/data/segmentation2/val_masks/'
    test_frame_path = '/data/segmentation2/test_frames/'
    test_mask_path = '/data/segmentation2/test_masks/'

    train_image_generator = train_datagen.flow_from_directory(train_frame_path, batch_size=BATCH_SIZE)
    train_mask_generator = train_datagen.flow_from_directory(train_mask_path, batch_size=BATCH_SIZE)
    val_image_generator = val_datagen.flow_from_directory(val_frame_path, batch_size=BATCH_SIZE)
    val_mask_generator = val_datagen.flow_from_directory(val_mask_path, batch_size=BATCH_SIZE)
    test_image_generator = val_datagen.flow_from_directory(test_frame_path, batch_size=BATCH_SIZE)
    test_mask_generator = val_datagen.flow_from_directory(test_mask_path, batch_size=BATCH_SIZE)

    # train_generator = zip(train_image_generator, train_mask_generator)
    # val_generator = zip(val_image_generator, val_mask_generator)
    # train_gen = data_gen(train_frame_path, train_mask_path, batch_size=BATCH_SIZE)
    # val_gen = data_gen(val_frame_path, val_mask_path, batch_size=BATCH_SIZE)
    # test_gen = data_gen(test_frame_path, test_mask_path, batch_size=BATCH_SIZE)

    NO_OF_TRAINING_IMAGES = len(os.listdir(train_frame_path))
    NO_OF_VAL_IMAGES = len(os.listdir(val_frame_path))
    NO_OF_TEST_IMAGES = len(os.listdir(test_frame_path))
    NO_OF_EPOCHS = 25
    weights_path = './weights_path/'
    m = unet()
    history = m.fit_generator(train_gen, epochs=NO_OF_EPOCHS, steps_per_epoch=(NO_OF_TRAINING_IMAGES // BATCH_SIZE),
                              validation_data=val_gen, validation_steps=(NO_OF_VAL_IMAGES // BATCH_SIZE))

    # scores = m.predict_generator(test_frame_path, NO_OF_TEST_IMAGES // BATCH_SIZE, workers=5)
    score = m.evaluate_generator(test_gen, NO_OF_TEST_IMAGES // BATCH_SIZE)
    print("Loss: ", score[0], "Accuracy: ", score[1])

# print(history.history.keys())
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()

# https://drive.google.com/file/d/1NB6ofDyoW5gfcC8q2512loCz0n8EZEIzoq/view?usp=sharing
