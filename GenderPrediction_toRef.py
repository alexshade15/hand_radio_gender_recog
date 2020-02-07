# from sklearn.preprocessing import LabelBinarizer
import cv2
import sys
import os
import utility

if __name__ == "__main__":
    architecture = sys.argv[1]
    unlock = sys.argv[2]
    weights = sys.argv[3]
    folder = sys.argv[4]
    index = int(sys.argv[5])

    if "validation" in folder:
        csvPath = "/data/val.csv"
        a = 1
    else:
        csvPath = "/data/train.csv"
        a = 2

    f = open(csvPath, "r")
    f.readline()
    labels = set()
    dictLabs = {}
    for line in f:
        line_content = line.strip().split(",")
        dictLabs[line_content[0]] = line_content[a]

    m = utility.load_model(unlock, weights, architecture)
    n = os.listdir(folder)

    img = cv2.imread(folder + '/' + n[index])
    img = cv2.resize(img / 255., (512, 512))
    img = img.reshape(512, 512, 3)

    number, ext = n[index].split(".")
    print("Image:", n[index])
    print("Label:", dictLabs[number])

    print("Prediction:", m.predict(img, verbose=1))
