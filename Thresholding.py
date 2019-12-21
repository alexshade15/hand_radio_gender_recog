import cv2
import os
from matplotlib import pyplot as plt


def thresholda(name):
    ''' Thresholds the radiograpys in order to obtain the masks to use in the unet training '''
    if not (os.path.isfile('./temp/masks' + str(name))):
        if os.path.isfile("./data/toTrain/" + str(name)):
            print(name)
            image = cv2.imread(filename="./data/toTrain/" + str(name))
            cv2.namedWindow(winname="Grayscale Image", flags=cv2.WINDOW_NORMAL)
            histogram = cv2.calcHist(images=[image], channels=[0], mask=None, histSize=[256], ranges=[0, 256])

            ok = 2
            while ok == 2:
                plt.figure()
                plt.title("Grayscale Histogram")
                plt.xlabel("grayscale value")
                plt.ylabel("pixels")
                plt.xlim([0, 256])
                plt.plot(histogram)
                plt.show()

                k = 11
                t = int(input('Enter the threshold: '))
                blur = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(src=blur, ksize=(k, k), sigmaX=0)
                (t, maskLayer) = cv2.threshold(src=blur, thresh=t, maxval=255, type=cv2.THRESH_BINARY)
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
                    cv2.imwrite('./temp/masks' + str(name), myMask)
                    cv2.imwrite('./temp/frames' + str(name), image)
                    print('./temp/frames' + str(name))
                    print(os.listdir('./temp/'))
                    print(os.getcwd())
        else:
            print(name, "not found!")


if __name__ == "__main__":
    k = os.listdir("./data/toTrain/")

    for elem in k:
        thresholda(elem)
