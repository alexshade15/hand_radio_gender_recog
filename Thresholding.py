import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import transform
from tensorflow.keras import Model
from tensorflow.keras.models import load_model


def plot_histogram(histogram, name):
    plt.figure(name.split(".")[0])
    plt.title(name + " - Grayscale Histogram")
    plt.xlabel("grayscale value")
    plt.ylabel("pixels")
    plt.xlim([0, 256])
    plt.plot(histogram)
    plt.show()


def resize(image):
    scale_percent = 80  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


def get_grad(cam_model, gap_weights, img):
    # test_img = cv2.imread(img_path + i).reshape(1, 512, 512, 3)

    features, results = cam_model.predict(img)
    cam_output = np.dot(features, gap_weights).reshape(16, 16)
    heatmap = np.maximum(cam_output, 0)
    heatmap /= np.max(heatmap)
    heatmap = np.uint8(255 * heatmap)
    heatmap = transform.resize(heatmap, (512, 512), preserve_range=True)
    return heatmap


def do_grad_threshold(image_path, dest_path, model_name):
    images = os.listdir(image_path)

    model = load_model(model_name)
    gap_weights = model.layers[-1].get_weights()[0]
    cam_model = Model(inputs=model.input,
                      outputs=(model.layers[-3].output,
                               model.layers[-1].output))

    for img in images:
        if "DS" not in img:
            image = cv2.imread(image_path + img)
            heatmap = get_grad(cam_model, gap_weights, image.reshape(1, 512, 512, 3))
            heatmap = heatmap.astype(np.uint8)
            (t, mask) = cv2.threshold(src=heatmap, thresh=55, maxval=255, type=cv2.THRESH_BINARY)
            mask = cv2.bitwise_not(mask)
            result = cv2.bitwise_and(image, image, mask=mask)
            cv2.imwrite(dest_path + img, result)


# do_grad_threshold("/Users/alex/Desktop/reduced1_handset/validation2/",
#                   "/Users/alex/Desktop/heatmap_dest/",
#                   "/Users/alex/Downloads/CAM.h5")


def thresholda(origin_path, dest_path, name):
    ''' Thresholds the radiograpys in order to obtain the masks to use in the unet training '''
    if not (os.path.isfile(dest_path + "masks/" + str(name))):
        if os.path.isfile(origin_path + str(name)):
            image = cv2.imread(filename=origin_path + str(name))
            # cv2.namedWindow(winname="Grayscale Image", flags=cv2.WINDOW_NORMAL)
            histogram = cv2.calcHist(images=[image], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
            plot_histogram(histogram, name)
            ok = 2
            while ok >= 2:
                if ok == 3:
                    plot_histogram(histogram, name)
                k = 11
                if ok >= 51:
                    t = ok
                else:
                    t = int(input('Enter the threshold: '))
                blur = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(src=blur, ksize=(k, k), sigmaX=0)
                # clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(64, 64))
                # blur = clahe.apply(blur)
                (t, maskLayer) = cv2.threshold(src=blur, thresh=t, maxval=255, type=cv2.THRESH_BINARY)
                my_mask2 = cv2.merge(mv=[maskLayer, maskLayer, maskLayer])

                contours, hierarchy = cv2.findContours(maskLayer, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                c = max(contours, key=cv2.contourArea)
                c_len = cv2.arcLength(c, True)

                for contour in contours:
                    contour_len = cv2.arcLength(contour, True)
                    if contour_len < c_len:
                        cv2.drawContours(my_mask2, [c], -1, 0, -1)
                my_mask2 = cv2.bitwise_not(my_mask2)
                my_mask = cv2.bitwise_and(my_mask2, my_mask2, mask=maskLayer)
                while True:

                    cv2.imshow("Before", resize(image))
                    cv2.imshow("After", resize(my_mask))
                    if cv2.waitKey(33) == 27:
                        break
                cv2.waitKey(delay=10)
                ok = int(input("ok? y-1, n/r-2, n/r/h-3: "))
                if ok == 1 or ok == -1 or ok == -2:
                    # if ok == -2:
                    #     cv2.imwrite(dest_path + 'masks/' + "temp_" + str(t) + "_" + str(name), my_mask)
                    # else:
                    #     cv2.imwrite(dest_path + 'masks/' + str(name), my_mask)
                    # cv2.imwrite(dest_path + 'frames/' + str(name), image)
                    print(dest_path + 'frames/' + str(name))
                    print(os.listdir(dest_path))
                    print(os.getcwd())
                    if ok == -1:
                        return -1
                    if ok == -2:
                        return -2
        else:
            print(name, "not found!")

if __name__ == "__main__":
    origin_path = "/Users/alex/Desktop/heatmaps/"

    # list_mod_post_threshold = [14392, 1804, 12091, 1810, 8834, 6831, 1379, 1476, 1497]
    # list_also_bad_unet_segmentation = [1388, 1402, 1406, 1407, 1418, 1426, 1469, 1518, 1541, 1559, 1596, 1601, 1701,
    #                                    1826, 1840]
    # definitivamente = [1431, 1494]
    # only_unet = [1489]

    # for image_name in image_list:
    #     image = cv2.imread(filename=path+str(image_name)+".png")
    #     histogram = cv2.calcHist(images=[image], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
    #
    #     plot_histogram(histogram, str(image_name))
    #     cv2.imshow(str(image_name), image)

    # origin_path = "/Users/alex/Desktop/new_normalized_training/"
    dest_path = "/Users/alex/Desktop/heatmap_dest/"
    k = os.listdir(origin_path)
    for index, elem in enumerate(k):
        # k[index] = str(elem) + ".png"
        if "hist" in elem:
            k.remove(elem)

    print(len(k))
    for elem in k:
        mode = thresholda(origin_path, dest_path, elem)
        if mode == -1:
            break
        if mode == -2:
            mode = thresholda(origin_path, dest_path, elem)
        if mode == -1:
            break