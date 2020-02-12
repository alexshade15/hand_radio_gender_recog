import model
import os
import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt


def plot_sample(X, y, preds, binary_preds, ix=0):
    has_mask = y[ix].max() > 0

    fig, ax = plt.subplots(1, 4, figsize=(20, 10))
    ax[0].imshow(X[ix, ..., 0], cmap='seismic')
    if has_mask:
        ax[0].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[0].set_title('Test')

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


def generateMasks(path, dest, folder, w_name):
    ''' folder: folder that contains radiograpy
        w_name: .h5 file
        generates in masked_normalized folder a mask for each radiograpy in folder'''

    m = model.unet()
    m.load_weights(w_name)

    images_folder = path + folder
    images = os.listdir(images_folder)
    i = 1
    num_images = len(images)

    for image in images:
        name, ext = image.split(".")

        if ext == "png":
            print("\n\n", images_folder + image, "---", str(i) + "/" + str(num_images), "---",
                  "%" + str(int(100 * i / num_images)))
            i += 1
            img_input = cv2.imread(images_folder + image, cv2.IMREAD_GRAYSCALE) / 255.
            img_input = cv2.resize(img_input, (512, 512))
            img_input = img_input.reshape(1, 512, 512, 1)

            img_pred = m.predict(img_input, verbose=1)
            img_mask = (img_pred > 0.3).astype(np.uint8)

            # from the prediction, takes only the biggest shapes. It is supposed to be the hand.
            mask = img_mask.squeeze() * 255
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            c = max(contours, key=cv2.contourArea)
            my_mask = cv2.merge(mv=[mask, mask, mask])
            clen = cv2.arcLength(c, True)
            for contour in contours:
                contourlen = cv2.arcLength(contour, True)
                if contourlen < clen:
                    cv2.drawContours(my_mask, [c], -1, 0, -1)
            my_mask = cv2.bitwise_not(my_mask)
            out_mask = cv2.bitwise_and(my_mask, my_mask, mask=mask)
            # out_mask = imFill(out_mask)  # fills holes in the hands if they occurs
            cv2.imwrite(dest + folder + image, out_mask)
            print('save to:' + dest + folder + image)


def imFill(path):
    ''' Fills the masks's holes'''
    images = os.listdir(path)
    for image in images:
        img = cv2.imread(path + image, 0)
        im_floodfill = img.copy()
        h, w = img.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(im_floodfill, mask, (0, 0), 255)
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        filled_img = img | im_floodfill_inv
        cv2.imwrite(path + image, filled_img)


def maskApply(imgPath, maskPath, dataset_type, dest):
    ''' Apply the masks from maskPath to the radiograpys from imgPath, obtaining the masked radiographys'''

    imgPath = imgPath + dataset_type
    maskPath = maskPath + dataset_type
    img_list = os.listdir(imgPath)
    mask_list = os.listdir(maskPath)
    # masked_list = os.listdir('/data/reduced_handset/' + dataset_type)
    for img in img_list:
        print("\n\nImage name: ", img)
        mask_name = img
        if mask_name in mask_list:  # and not(mask_name in masked_list):
            image = cv2.imread(imgPath + img)
            mask = cv2.imread(maskPath + mask_name, cv2.IMREAD_GRAYSCALE)
            res = cv2.bitwise_and(image, image, mask=mask)
            print("masked")
            b, g, r = cv2.split(res)
            clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(64, 64))
            r = clahe.apply(r)
            g = clahe.apply(g)
            b = clahe.apply(b)
            print("clahed")
            image = cv2.merge((r, g, b))
            cv2.imwrite(dest + dataset_type + img, image)
            print("saved")
        else:
            if not (mask_name in mask_list):
                print("Errore, " + img + " non presente nelle maschere.", file=sys.stderr)


def test():
    m = model.unet()
    m.load_weights('ultiSeg92.h5')

    test_folder_frame = './data/normalized/training'
    # test_folder_masks = './data/normalized/training'

    n1 = os.listdir(test_folder_frame)
    i = 1000
    # name, ext = n1[i].split(".")

    train_frame_img = cv2.imread(test_folder_frame + '/' + n1[i], cv2.IMREAD_GRAYSCALE) / 255.
    train_frame_img = cv2.resize(train_frame_img, (512, 512))
    train_frame_img = train_frame_img.reshape(1, 512, 512, 1)

    preds_train = m.predict(train_frame_img, verbose=1)
    preds_train_t = (preds_train > 0.3).astype(np.uint8)
    plot_sample(train_frame_img, train_frame_img, preds_train, preds_train_t)


generateMasks("/data/test_normalized/", "/data/test_masks/", "", "pesi_unet_983_8feb.h5")
imFill("/data/test_masks/")
maskApply("/data/test_normalized/", "/data/test_masks/", "", "/data/test_handset/")
