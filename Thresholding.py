import cv2
import sys
import os
from matplotlib import pyplot as plt
import numpy as np


def get_largest(im, n):
    # Find contours of the shape
    contours, _ = cv2.findContours(im.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Cycle through contours and add area to array
    areas = []
    for c in contours:
        areas.append(cv2.contourArea(c))

    # Sort array of areas by size
    sorted_areas = sorted(zip(areas, contours), key=lambda x: x[0], reverse=True)

    if sorted_areas and len(sorted_areas) >= n:
        # Find nth largest using data[n-1][1]
        return sorted_areas[n - 1][1]
    else:
        return None


def thresholda(name):
    if not (os.path.isfile('./data/dataForSegmentation/masks' + str(name) + ".png")) and not (
            os.path.isfile('./temp/masks' + str(name) + ".png")):
        if os.path.isfile("./data/normalized/training/" + str(name) + ".png"):
            image = cv2.imread(filename="./data/normalized/training/" + str(name) + ".png")
            cv2.namedWindow(winname="Grayscale Image", flags=cv2.WINDOW_NORMAL)
            histogram = cv2.calcHist(images=[image], channels=[0], mask=None, histSize=[256], ranges=[0, 256])

            ok = 2
            while ok == 2:
                # configure and draw the histogram figure
                plt.figure()
                plt.title("Grayscale Histogram")
                plt.xlabel("grayscale value")
                plt.ylabel("pixels")
                plt.xlim([0, 256])  # <- named arguments do not work here
                plt.plot(histogram)  # <- or here
                plt.show()

                k = 11
                t = int(input('Enter the threshold: '))
                blur = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(src=blur, ksize=(k, k), sigmaX=0)
                (t, maskLayer) = cv2.threshold(src=blur, thresh=t, maxval=255, type=cv2.THRESH_BINARY)
                # myMask = cv2.merge(mv=[maskLayer, maskLayer, maskLayer])
                myMask2 = cv2.merge(mv=[maskLayer, maskLayer, maskLayer])

                contours, hierarchy = cv2.findContours(maskLayer, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                c = max(contours, key=cv2.contourArea)
                cLen = cv2.arcLength(c, True)

                for contour in contours:
                    contourLen = cv2.arcLength(contour, True)
                    if contourLen < cLen:
                        cv2.drawContours(myMask2, [c], -1, 0, -1)
                myMask2 = cv2.bitwise_not(myMask2)
                myMask = cv2.bitwise_and(myMask2, myMask2, mask=maskLayer)
                cv2.imshow("After", myMask)
                cv2.waitKey(delay=10)
                ok = int(input("ok? y-1, n/r-2: "))
                if ok == 1:
                    cv2.imwrite('./temp/masks' + str(name) + ".png", myMask)
                    cv2.imwrite('./temp/frames' + str(name) + ".png", image)
        else:
            print(name, "not found!")


if __name__ == "__main__":
    # k = [2482, 2500, 2506, 2522, 2541, 2564, 2571, 2583, 2589, 2591, 2638, 2668, 2669, 2681, 2697, 2710, 2712, 2725,
         2728, 2729, 2743, 2750, 2780, 2828, 2850]

    for elem in k:
        thresholda(elem)

    # filename = "./data/norm134alized/training/13358.png"
    # path = "./data/normalized/training/"
    # frames = os.listdir(path)
    #
    # stop = False
    # for index, frame in enumerate(frames):
    #     if not (os.path.isfile('./data/dataForSegmentation/masks' + frame)) and \
    #             not (os.path.isfile('./dataset1/masks' + frame)) and \
    #             not (os.path.isfile('./dataset2/masks' + frame)):
    #         print(frame)
    #         image = cv2.imread(filename=path + frame)  # , flags=cv2.IMREAD_GRAYSCALE)
    #
    #         # display the image
    #         cv2.namedWindow(winname="Grayscale Image", flags=cv2.WINDOW_NORMAL)
    #         cv2.imshow(winname="Grayscale Image", mat=image)
    #         # print("wait")
    #         # cv2.waitKey(delay=2000)
    #         # print("ok")
    #         # create the histogram
    #         histogram = cv2.calcHist(images=[image], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
    #         ok = 2
    #         while ok == 2:
    #             # configure and draw the histogram figure
    #             plt.figure()
    #             plt.title("Grayscale Histogram")
    #             plt.xlabel("grayscale value")
    #             plt.ylabel("pixels")
    #             plt.xlim([0, 256])  # <- named arguments do not work here
    #             plt.plot(histogram)  # <- or here
    #             plt.show()
    #
    #             k = 11
    #             t = 37
    #             # k = int(input('Enter the kernel size: '))
    #             t = int(input('Enter the threshold: '))
    #             # print("kernel:", k, "threshold:", t)
    #             # get filename, kernel size, and threshold value from command line
    #
    #             # blur and grayscale before thresholding
    #             blur = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
    #             # ------- CLAHE -------
    #             # cv2.imshow(winname="CLAHE", mat=blur)
    #             # clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(64, 64))
    #             # blur = clahe.apply(blur)
    #             blur = cv2.GaussianBlur(src=blur, ksize=(k, k), sigmaX=0)
    #
    #             # perform inverse binary thresholding
    #             (t, maskLayer) = cv2.threshold(src=blur, thresh=t, maxval=255, type=cv2.THRESH_BINARY)
    #             # print("Threshold:", t)
    #             # make a mask suitable for color images
    #             myMask = cv2.merge(mv=[maskLayer, maskLayer, maskLayer])
    #             myMask2 = cv2.merge(mv=[maskLayer, maskLayer, maskLayer])
    #             # cv2.namedWindow(winname="mask", flags=cv2.WINDOW_NORMAL)
    #             # cv2.imshow(winname="mask", mat=myMask)
    #
    #             # # use the mask to select the "interesting" part of the image
    #             # sel = cv2.bitwise_and(src1=image, src2=mask)
    #             #
    #             # # display the result
    #             # cv2.namedWindow(winname="selected", flags=cv2.WINDOW_NORMAL)
    #             # cv2.imshow(winname="selected", mat=sel)
    #
    #             contours, hierarchy = cv2.findContours(maskLayer, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #             c = max(contours, key=cv2.contourArea)
    #             cLen = cv2.arcLength(c, True)
    #
    #             for contour in contours:
    #                 contourLen = cv2.arcLength(contour, True)
    #                 if contourLen < cLen:
    #                     # print(contourLen)
    #                     # cv2.drawContours(mask, c, -1, 100, 3)
    #                     # cv2.fillPoly(mask, contour, color=(0, 0, 0))
    #                     cv2.drawContours(myMask2, [c], -1, 0, -1)
    #             myMask2 = cv2.bitwise_not(myMask2)
    #             myMask = cv2.bitwise_and(myMask2, myMask2, mask=maskLayer)
    #             cv2.imshow("After", myMask)
    #
    #             cv2.waitKey(delay=10)
    #
    #             ok = int(input("ok? y-1, n/r-2, s-3: "))
    #             if ok == 1 or ok == 3:
    #                 cv2.imwrite('./dataset2/masks' + str(frame), myMask)
    #                 cv2.imwrite('./dataset2/frames' + str(frame), image)
    #         if ok == 3:
    #             break
