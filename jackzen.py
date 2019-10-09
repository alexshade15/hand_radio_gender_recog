import tensorflow
from tensorflow.compat.v2.keras.layers import *
from tensorflow.compat.v2.keras.models import Model
from tensorflow.compat.v2.keras.optimizers import *
from keras.preprocessing.image import ImageDataGenerator
import os
import cv2
import random
import numpy as np
from matplotlib import pyplot as plt


def data_gen(img_folder, mask_folder, batch_size):
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
    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
    m = Model(inputs=inputs, outputs=conv10)
    m.compile(optimizer=SGD(lr=0.015), loss='binary_crossentropy', metrics=['accuracy'])
    return m

train_datagen = ImageDataGenerator(rescale=1. / 255)
val_datagen = ImageDataGenerator()
BATCH_SIZE = 16
train_image_generator = train_datagen.flow_from_directory('/data/masked_hands/train_frames', batch_size=BATCH_SIZE)
train_mask_generator = train_datagen.flow_from_directory('/data/masked_hands/train_masks', batch_size=BATCH_SIZE)
val_image_generator = val_datagen.flow_from_directory('/data/masked_hands/val_frames', batch_size=BATCH_SIZE)
val_mask_generator = val_datagen.flow_from_directory('/data/masked_hands/val_masks', batch_size=BATCH_SIZE)
train_generator = zip(train_image_generator, train_mask_generator)
val_generator = zip(val_image_generator, val_mask_generator)
train_frame_path = '/data/masked_hands/train_frames/train'
train_mask_path = '/data/masked_hands/train_masks/train'
val_frame_path = '/data/masked_hands/val_frames/val'
val_mask_path = '/data/masked_hands/val_masks/val'
train_gen = data_gen(train_frame_path, train_mask_path, batch_size=BATCH_SIZE)
val_gen = data_gen(val_frame_path, val_mask_path, batch_size=BATCH_SIZE)
NO_OF_TRAINING_IMAGES = len(os.listdir('/data/masked_hands/train_frames/train'))
NO_OF_VAL_IMAGES = len(os.listdir('/data/masked_hands/val_frames/val'))
NO_OF_EPOCHS = 25
weights_path = './weights_path/'
m = unet()
history = m.fit_generator(train_gen, epochs=NO_OF_EPOCHS, steps_per_epoch=(NO_OF_TRAINING_IMAGES // BATCH_SIZE),
                          validation_data=val_gen, validation_steps=(NO_OF_VAL_IMAGES // BATCH_SIZE))

print(history.history.keys())
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
