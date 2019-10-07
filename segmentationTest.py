from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

import model
import os
import cv2
import random
import numpy as np
from matplotlib import pyplot as plt


def plot_sample(X, y, preds, binary_preds, ix=0):
    has_mask = y[ix].max() > 0

    fig, ax = plt.subplots(1, 4, figsize=(20, 10))
    ax[0].imshow(X[ix, ..., 0], cmap='seismic')
    if has_mask:
        ax[0].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[0].set_title('Seismic')

    ax[1].imshow(y[ix].squeeze())
    ax[1].set_title('Mask')

    ax[2].imshow(preds[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[2].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[2].set_title('Prediction')

    ax[3].imshow(binary_preds[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[3].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[3].set_title('Binary Prediction')


def predict(n, index, threshold):
    tfi = cv2.imread(test_folder_frame + '/' + n[index], cv2.IMREAD_GRAYSCALE) / 255.
    tfi = cv2.resize(tfi, (512, 512))
    tfi = tfi.reshape(1, 512, 512, 1)

    # train_masks_img = cv2.imread(test_folder_masks + '/' + "masks" + number + "." + ext, cv2.IMREAD_GRAYSCALE) / 255.
    # train_masks_img = cv2.resize(train_masks_img, (512, 512))
    # train_masks_img = train_masks_img.reshape(1, 512, 512, 1)

    pt = m.predict(tfi, verbose=1)
    ptt = (pt > threshold).astype(np.uint8)
    # plot_sample(train_frame_img, train_masks_img, preds_train, preds_train_t)
    plot_sample(tfi, tfi, pt, ptt)

    return ptt

if __name__ == "__main__":
    m = model.unet()
    m.load_weights('MoreData_SGD015_batch16_newIMG.h5')

    test_folder_frame = 'normalized/training'
    test_folder_masks = 'normalized/training'

    # test_folder_frame = 'masked_hands/train_frames/train'
    # test_folder_masks = 'masked_hands/train_masks/train'

    # test_folder_frame = 'dataForSegmentation'
    # test_folder_masks = 'dataForSegmentation'
    n1 = os.listdir(test_folder_frame)
    i = 1000
    # name, ext = n1[i].split(".")
    # _, number = name.split("s")

    train_frame_img = cv2.imread(test_folder_frame + '/' + n1[i], cv2.IMREAD_GRAYSCALE) / 255.
    train_frame_img = cv2.resize(train_frame_img, (512, 512))
    train_frame_img = train_frame_img.reshape(1, 512, 512, 1)

    # train_masks_img = cv2.imread(test_folder_masks + '/' + "masks" + number + "." + ext, cv2.IMREAD_GRAYSCALE) / 255.
    # train_masks_img = cv2.resize(train_masks_img, (512, 512))
    # train_masks_img = train_masks_img.reshape(1, 512, 512, 1)

    preds_train = m.predict(train_frame_img, verbose=1)
    preds_train_t = (preds_train > 0.3).astype(np.uint8)
    # plot_sample(train_frame_img, train_masks_img, preds_train, preds_train_t)
    plot_sample(train_frame_img, train_frame_img, preds_train, preds_train_t)

    # import os
    # path = "masked_hands/train_masks/train/"
    # for filename in os.listdir(path):
    #     number, _ = filename.split("_")
    #     dst = number + ".png"
    #     src = path + filename
    #     dst = path + dst
    #     os.rename(src, dst)
    
    # MODELS
    # DO    Model.h5
    # NO    Model_sgd.h5
    # NO    Model_sgd_lr0.01_batch1.h5
    # NO    Model_sgd_lr0.005.h5
    # NOT    MoreData_SGD01_batch4_stdIMG.h5
    # NO    MoreData_SGD01_batch8.h5
    # BEST?    MoreData_SGD01_batch16.h5
    # BEST!    MoreData_SGD015_batch16_newIMG.h5


    # import io
    # from PIL import Image
    #
    #
    # image = Image.open(io.BytesIO(preds_train_t))
    # image.save("imgTEST")
