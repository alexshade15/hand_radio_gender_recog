from PIL import Image
import glob

def make_square(im, min_size=256, fill_color=(0, 0, 0, 0)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGBA', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im


# photos = glob.glob("/Users/alex/Desktop/prova/images/mass/*.tif")
# photos = ["/Users/alex/Desktop/20586908_6c613a14b80a8591_MG_R_CC_ANON.png"]
photos = ["/Users/alex/Desktop/new_training/3752.jpg"]
# photos = glob.glob("/Users/alex/Desktop/new_training/*.jpg")
for photo in photos:
    name = photo.split("/")[-1]
    tmp = name.split(".")
    print(tmp[0], tmp[1])
    img = Image.open(photo)
    img = make_square(img)
    print("Squared")
    width = 512
    height = 512
    img = img.resize((width, height), Image.ANTIALIAS)  # best down-sizing filter
    print("Resized")
    img.save("/Users/alex/Desktop/new_normalized_training/" + tmp[0] + ".png", "PNG")
    print("Saved\n\n")

# import cv2
# import os
#
# print(os.getcwd())
# filename = '/Users/alex/Desktop/prova/images/mass/20586908_6c613a14b80a8591_MG_R_CC_ANON.tif'
# W = 1000.
# img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
# imgScale = W / img.shape[1]
# newX, newY = img.shape[1] * imgScale, img.shape[0] * imgScale
# newimg = cv2.resize(img, (int(newX), int(newY)))
#
# cv2.imwrite("resizeimg.jpg", newimg)
# cv2.imshow("Show by CV2", newimg)
# cv2.waitKey(0)
