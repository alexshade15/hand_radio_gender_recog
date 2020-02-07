import cv2
import os
from matplotlib import pyplot as plt


# import keyboard

def plot_histogram(histogram, name):
    plt.figure()
    plt.title(name + " - Grayscale Histogram")
    plt.xlabel("grayscale value")
    plt.ylabel("pixels")
    plt.xlim([0, 256])
    plt.plot(histogram)
    plt.show()


def thresholda(origin_path, dest_path, name):
    ''' Thresholds the radiograpys in order to obtain the masks to use in the unet training '''
    if not (os.path.isfile(dest_path + "masks/" + str(name))):
        if os.path.isfile(origin_path + str(name)):
            print(name)
            image = cv2.imread(filename=origin_path + str(name))
            # cv2.namedWindow(winname="Grayscale Image", flags=cv2.WINDOW_NORMAL)
            histogram = cv2.calcHist(images=[image], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
            plot_histogram(histogram, name)
            ok = 2
            while ok >= 2:
                if ok == 3:
                    plot_histogram(histogram, name)

                k = 11
                if ok > 10:
                    t = ok
                else:
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
                while True:
                    cv2.imshow("Before", image)
                    cv2.imshow("After", myMask)
                    if cv2.waitKey(33) == 27:
                        break
                cv2.waitKey(delay=10)
                ok = int(input("ok? y-1, n/r-2, n/r/h-3: "))
                if ok == 1 or ok == -1 or ok == -2:
                    if ok == -2:
                        cv2.imwrite(dest_path + 'masks/' + "temp_" + str(t) + "_" + str(name), myMask)
                    else:
                        cv2.imwrite(dest_path + 'masks/' + str(name), myMask)
                    cv2.imwrite(dest_path + 'frames/' + str(name), image)
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
    origin_path = "/Users/alex/Desktop/new_normalized_training/"
    dest_path = "/Users/alex/Desktop/new_groundtruth/"
    k = [3128, 3131, 3133, 3134, 3136, 3137, 3138, 3140, 3141, 3142, 3143]
    k += [3144, 3145, 3145, 3148, 3149, 3151, 3153, 3156, 3157, 3164, 3165]
    k += [3167, 3169, 3172, 3175, 3176, 3179, 3181, 3182, 3183, 3185, 3186]
    k += [3187, 3188, 3189, 3196, 3200, 3201, 3202, 3204, 3205, 3211, 3213]

    for index, elem in enumerate(k):
        k[index] = str(elem) + ".png"

    print(len(k))
    for elem in k:
        mode = thresholda(origin_path, dest_path, elem)
        if mode == -1:
            break
        if mode == -2:
            mode = thresholda(origin_path, dest_path, elem)
        if mode == -1:
            break
