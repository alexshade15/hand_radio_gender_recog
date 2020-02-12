from tensorflow.compat.v2.keras.optimizers import *

import os
import sys
import utility


def main(epoch=10, batch_size=64, unlock=False, weights=None, optimizer=(SGD(), "SGD"), my_lr=0.001, my_momentum=0.9,
         my_nesterov=False, my_decay=0.0, log_name="unnamed"):
    csv_path = "/data/unified.csv"
    train_path = '/data/original_r2_handset/training/'
    val_path = '/data/original_r2_handset/validation1/'
    test_path = '/data/original_r2_handset/validation2/'

    utility.doTraining(epoch, batch_size, optimizer, my_lr, my_momentum, my_nesterov, my_decay, unlock, weights,
                       csv_path, os.listdir(train_path), train_path, os.listdir(val_path), val_path,
                       os.listdir(test_path), test_path, log_name)


if __name__ == "__main__":
    try:
        epoch = int(sys.argv[1])
    except IndexError:
        epoch = 50
    try:
        batch_size = int(sys.argv[2])
    except IndexError:
        batch_size = 64
    try:
        unlock = sys.argv[3]
    except IndexError:
        unlock = False
    try:
        weights = sys.argv[4]
    except IndexError:
        weights = None
    print("epoch: %d, batch_size: %d, unlock: %s, weights: %s \n\n" % (epoch, batch_size, unlock, weights))

    for i in [15]:
        print("epochs: {}, bs: {}, unlock: {}, pesi: {}, opt: {}, lr: {}, mom: {}, nest: {}, dec: {}".format(
            epoch, batch_size, unlock, weights, utility.optimizers[i], utility.lrs[i], utility.moms[i],
            utility.nesterovs[i], utility.decays[i]))

        main(epoch, batch_size, unlock, weights, utility.optimizers[i], utility.lrs[i], utility.moms[i],
             utility.nesterovs[i], utility.decays[i], "original_reduced_Close_sgd29_" + str(i))
        print("Training succesfully")

# def test_vgg16():
#     unlock = True
#     weights = "path"
#     mode = 0 # orifinal architecture
#     model = load_model(unlock, weights, mode)
#
#     trainCsvPath = "/data/train.csv"
#     valCsvPath = "/data/val.csv"
#     trainPath = '/data/handset/training/'
#     valPath = '/data/handset/validation1/'
#     testPath = '/data/handset/validation2/'
#
#     n1 = os.listdir(trainPath)
#     i = 1000
#     # name, ext = n1[i].split(".")
#
#     train_img = cv2.imread("path" + '/' + n1[i])
#     train_img = cv2.resize(train_img / 255., (512, 512))
#     train_img = train_img.reshape(512, 512, 3)
#
#     print(model.predict(train_img, verbose=1))
