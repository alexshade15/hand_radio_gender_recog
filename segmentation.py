from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

import os
import cv2
import random
import numpy as np
from matplotlib import pyplot as plt

os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'

train_datagen = ImageDataGenerator(
    rescale=1. / 255)
# ,
# shear_range=0.2,
# zoom_range=0.2,
# horizontal_flip=True)

val_datagen = ImageDataGenerator()  # rescale=1. / 255)

BATCH_SIZE = 16

# NORMALLY 4/8/16/32)
train_image_generator = train_datagen.flow_from_directory('masked_hands/train_frames', batch_size=BATCH_SIZE)
train_mask_generator = train_datagen.flow_from_directory('masked_hands/train_masks', batch_size=BATCH_SIZE)
val_image_generator = val_datagen.flow_from_directory('masked_hands/val_frames', batch_size=BATCH_SIZE)
val_mask_generator = val_datagen.flow_from_directory('masked_hands/val_masks', batch_size=BATCH_SIZE)

train_generator = zip(train_image_generator, train_mask_generator)
val_generator = zip(val_image_generator, val_mask_generator)


def data_gen(img_folder, mask_folder, batch_size):
    c = 0
    n1 = os.listdir(img_folder)  # List of training images
    # n2 = os.listdir(mask_folder)  # List of training images
    random.shuffle(n1)

    while True:
        img = np.zeros((batch_size, 512, 512, 1)).astype('float')
        mask = np.zeros((batch_size, 512, 512, 1)).astype('float')

        for i in range(c, c + batch_size):  # initially from 0 to 16, c = 0.

            # print(n1[i])

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


train_frame_path = 'masked_hands/train_frames/train'
train_mask_path = 'masked_hands/train_masks/train'

val_frame_path = 'masked_hands/val_frames/val'
val_mask_path = 'masked_hands/val_masks/val'

# Train the model
train_gen = data_gen(train_frame_path, train_mask_path, batch_size=BATCH_SIZE)
val_gen = data_gen(val_frame_path, val_mask_path, batch_size=BATCH_SIZE)

NO_OF_TRAINING_IMAGES = len(os.listdir('masked_hands/train_frames/train'))
NO_OF_VAL_IMAGES = len(os.listdir('masked_hands/val_frames/val'))

NO_OF_EPOCHS = 50

weights_path = './weights_path/'

m = model.unet()
# opt = Adam(lr=1E-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
# m.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])

# checkpoint = ModelCheckpoint(weights_path, monitor='accuracy', verbose=1, save_best_only=True, mode='max')

# csv_logger = CSVLogger('./log.out', append=True, separator=';')

# earlystopping = EarlyStopping(monitor='accuracy', verbose=1, min_delta=0.01, patience=3, mode='max')

# callbacks_list = [checkpoint, csv_logger, earlystopping]

history = m.fit_generator(train_gen, epochs=NO_OF_EPOCHS, steps_per_epoch=(NO_OF_TRAINING_IMAGES // BATCH_SIZE),
                          validation_data=val_gen, validation_steps=(NO_OF_VAL_IMAGES // BATCH_SIZE))
# callbacks=callbacks_list)
m.save('MoreData_SGD015_batch16_newIMG.h5')

print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
