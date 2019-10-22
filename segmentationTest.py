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


def generateMasks(folder):
    m = model.unet()
    m.load_weights('/data/ultiSeg.h5')

    images_folder = '/data/normalized/' + folder
    images = os.listdir(images_folder)
    i = 1
    num_images = len(images)

    for image in images:
        name, ext = image.split(".")

        if ext == "png":
            print("\n\n", image, "---", str(i) + "/" + str(num_images), "---", "%" + str(int(100 * i / num_images)))
            i += 1
            img_input = cv2.imread(images_folder + '/' + image, cv2.IMREAD_GRAYSCALE) / 255.
            img_input = cv2.resize(img_input, (512, 512))
            img_input = img_input.reshape(1, 512, 512, 1)

            img_pred = m.predict(img_input, verbose=1)
            img_mask = (img_pred > 0.3).astype(np.uint8)

            mask = img_mask.squeeze() * 255
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            c = max(contours, key=cv2.contourArea)
            # myMask2 = cv2.merge(mv=[mask, mask, mask])
            my_mask = cv2.merge(mv=[mask, mask, mask])
            clen = cv2.arcLength(c, True)
            for contour in contours:
                contourlen = cv2.arcLength(contour, True)
                if contourlen < clen:
                    cv2.drawContours(my_mask, [c], -1, 0, -1)
            my_mask = cv2.bitwise_not(my_mask)
            out_mask = cv2.bitwise_and(my_mask, my_mask, mask=mask)
            cv2.imwrite('/data/normalized_masks/' + folder + '/' + image, out_mask)


def maskApply(imgPath, maskPath, dataset_type):
    import sys
    img_list = os.listdir(imgPath)
    mask_list = os.listdir(maskPath)
    for img in img_list:
        number, _ = img.split(".")
        mask_name = "masks" + number + ".png"
        if mask_name in mask_list:
            image = cv2.imread(imgPath + img)
            mask = cv2.imread(maskPath + mask_name, cv2.IMREAD_GRAYSCALE)
            res = cv2.bitwise_and(image, image, mask=mask)
            clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(64, 64))
            image = clahe.apply(res)
            cv2.imwrite('/data/handset/' + dataset_type + '/' + img, image)
        else:
            print("Errore, " + img + " non presente nelle maschere.", file=sys.stderr)


if __name__ == "__main__":
    m = model.unet()
    m.load_weights('MoreData_SGD015_batch16_newIMG.h5')

    test_folder_frame = './data/normalized/training'
    test_folder_masks = './data/normalized/training'

    # test_folder_frame = 'masked_hands/train_frames/train'
    # test_folder_masks = 'masked_hands/train_masks/train'

    # test_folder_frame = 'dataForSegmentation'
    # test_folder_masks = 'dataForSegmentation'
    n1 = os.listdir(test_folder_frame)
    i = 1000
    name, ext = n1[i].split(".")
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

    # cv2.imwrite('./temp/' + n1[i], preds_train_t.squeeze() * 255)

    # MODELS
    # DO    Model.h5
    # NO    Model_sgd.h5
    # NO    Model_sgd_lr0.01_batch1.h5
    # NO    Model_sgd_lr0.005.h5
    # NOT    MoreData_SGD01_batch4_stdIMG.h5
    # NO    MoreData_SGD01_batch8.h5
    # BEST?    MoreData_SGD01_batch16.h5
    # BEST!    MoreData_SGD015_batch16_newIMG.h5
