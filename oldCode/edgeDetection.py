import cv2, sys, numpy as np
from matplotlib import pyplot as plt

# read command-line arguments
filename = "1377.png"
lo = 270
hi = 300

# load and display original image as grayscale
image = cv2.imread(filename=filename, flags=cv2.IMREAD_GRAYSCALE)
cv2.namedWindow(winname="original", flags=cv2.WINDOW_NORMAL)
cv2.imshow(winname="original", mat=image)

# clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(64, 64))
# image = clahe.apply(image)
# cv2.imshow(winname="CLAHE", mat=image)

# # create the histogram
# histogram = cv2.calcHist(images=[image],
#                          channels=[0],
#                          mask=None,
#                          histSize=[256],
#                          ranges=[0, 256])
#
# # configure and draw the histogram figure
# plt.figure()
# plt.title("Grayscale Histogram")
# plt.xlabel("grayscale value")
# plt.ylabel("pixels")
# plt.xlim([0, 256])  # <- named arguments do not work here
# plt.plot(histogram)  # <- or here
# plt.show()

edges = cv2.Canny(image=image, threshold1=lo, threshold2=hi)

# display edges
cv2.namedWindow(winname="edges", flags=cv2.WINDOW_NORMAL)
cv2.imshow(winname="edges", mat=edges)
cv2.waitKey(delay=5000)
