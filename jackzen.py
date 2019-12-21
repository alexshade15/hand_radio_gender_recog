import tensorflow
from tensorflow.compat.v2.keras.layers import *
from tensorflow.compat.v2.keras.optimizers import *
import os
import cv2
import random
import numpy as np
import traceback
import sys
import model


def data_gen(img_folder, mask_folder, batch_size, aug=None):
    c = 0
    n1 = os.listdir(img_folder)  # List of training images
    random.shuffle(n1)
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


def myGrid(epoch=50, bs=4):

    learn_rate = [0.001, 0.01, 0.0001, 0.1, 0.2, 0.3]
    momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]

    train_frame_path = '/data/segmentation2/train_frames/train/'
    train_mask_path = '/data/segmentation2/train_masks/train/'
    val_frame_path = '/data/segmentation2/val_frames/val/'
    val_mask_path = '/data/segmentation2/val_masks/val/'
    test_frame_path = '/data/segmentation2/test_frames/test/'
    test_mask_path = '/data/segmentation2/test_masks/test/'

    NO_OF_EPOCHS = epoch
    BATCH_SIZE = bs

    try:
        for lr in learn_rate:
            for mom in momentum:
                m = model.unet()
                optimizer = "SGD"
                m.compile(optimizer=SGD(learning_rate=lr, momentum=mom, nesterov=True), loss='binary_crossentropy',
                          metrics=['accuracy'])

                train_gen = data_gen(train_frame_path, train_mask_path, batch_size=BATCH_SIZE)
                val_gen = data_gen(val_frame_path, val_mask_path, batch_size=BATCH_SIZE)
                test_gen = data_gen(test_frame_path, test_mask_path, batch_size=BATCH_SIZE)
                NO_OF_TRAINING_IMAGES = len(os.listdir(train_frame_path))
                NO_OF_VAL_IMAGES = len(os.listdir(val_frame_path))
                NO_OF_TEST_IMAGES = len(os.listdir(test_frame_path))

                history = m.fit_generator(train_gen, epochs=NO_OF_EPOCHS,
                                          steps_per_epoch=(NO_OF_TRAINING_IMAGES // BATCH_SIZE),
                                          validation_data=val_gen, validation_steps=(NO_OF_VAL_IMAGES // BATCH_SIZE))
                score = m.evaluate_generator(test_gen, NO_OF_TEST_IMAGES // BATCH_SIZE)
                print("\n\nScore: " + str(score))
                print("train acc " + str(history.history['accuracy']))
                print("valid acc " + str(history.history['val_accuracy']))
                print("learningRate: ", lr, "\nmomentum: ", mom)
                m.save("./models_unet/seg" + "_opt:" + str(optimizer) + "_ep:" + str(NO_OF_EPOCHS) + "_bs:" + str(
                    BATCH_SIZE) + "_lr:" + str(lr) + "_mom:" + str(mom) + "_loss:" + str(score[1]) + "_acc:" + str(
                    score[0]) + ".h5", "w+")
                f = open("./models_unet/seg" + "_opt:" + str(optimizer) + "_ep:" + str(NO_OF_EPOCHS) + "_bs:" + str(
                    BATCH_SIZE) + "_lr:" + str(lr) + "_mom:" + str(mom) + "_loss:" + str(score[1]) + "_acc:" + str(
                    score[0]) + ".txt", "w+")
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
        print(traceback.format_exc())
        print(sys.exc_info()[2])
        f = open("error_log_unet.txt", "w+")
        f.write(traceback.format_exc())
        f.write(str(sys.exc_info()[2]))
        f.close()


if __name__ == "__main__":
    myGrid(50, 4)
    print("Training went well!")
