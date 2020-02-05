from PIL import Image
import glob
import os

def make_square(im, min_size=256, fill_color=(0, 0, 0, 0)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGBA', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im

def main(ori_folder, dest_folder):
    photos = glob.glob(ori_folder + "*.jpg")
    dest = os.listdir(dest_folder)
    for photo in photos:
        name = photo.split("/")[-1].split(".")[0]
        name = name + ".png"
        if (name) not in dest:
            print(name)
            img = Image.open(photo)
            img = make_square(img)
            width = 512
            height = 512
            img = img.resize((width, height), Image.ANTIALIAS)  # best down-sizing filter
            img.save(dest_folder + name, "PNG")
# photos = glob.glob("/Users/alex/Desktop/prova/images/mass/*.tif")
# photos = ["/Users/alex/Desktop/20586908_6c613a14b80a8591_MG_R_CC_ANON.png"]
# photos = ["/Users/alex/Desktop/new_training/3752.jpg"]
# photos = glob.glob("/Users/alex/Desktop/new_training/*.jpg")

# import cv2
# import os

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
