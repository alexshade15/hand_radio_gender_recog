from tensorflow.keras.optimizers import *

import os
import sys
import utility


def main(epoch=10, batch_size=64, unlock=False, weights=None, optimizer=(SGD(), "SGD"), my_lr=0.001, my_momentum=0.9,
         my_nesterov=False, my_decay=0.0, log_name="unnamed"):
    csv_path = "/Users/alex/Desktop/full.csv"
    train_path = '/Users/alex/Desktop/bone age/training/'
    val_path = '/Users/alex/Desktop/bone age/validation/boneage-validation-dataset-1/'
    test_path = '/Users/alex/Desktop/bone age/validation/boneage-validation-dataset-2/'

    utility.do_training(epoch, batch_size, optimizer, my_lr, my_momentum, my_nesterov, my_decay, unlock, weights,
                        csv_path, os.listdir(train_path)[:100], train_path, os.listdir(val_path)[:100], val_path,
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
        unlock = True
    try:
        weights = sys.argv[4]
    except IndexError:
        weights = None

    for i in [15]:
        print("epochs: {}, bs: {}, unlock: {}, pesi: {}, opt: {}, lr: {}, mom: {}, nest: {}, dec: {}".format(
            epoch, batch_size, unlock, weights, utility.optimizers[i], utility.lrs[i], utility.moms[i],
            utility.nesterovs[i], utility.decays[i]))

        main(epoch, batch_size, unlock, weights, utility.optimizers[i], utility.lrs[i], utility.moms[i],
             utility.nesterovs[i], utility.decays[i], "original_reduced_open_handVGG16_" + str(i))
        print("Training succesfully")