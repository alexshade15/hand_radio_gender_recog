from tensorflow.keras.optimizers import *

import os
import sys
import numpy as np
import kfold_new_indexes
import utility


def main(epoch=10, batch_size=64, unlock=False, weights=None, optimizer=(SGD(), "SGD"), my_lr=0.001, my_momentum=0.9,
         my_nesterov=False, my_decay=0.0, log_name="unnamed"):
    csv_path = "/data/new.csv"
    train_path = '/data/r_r_handset/training/'

    # kf = KFold(n_splits=5, shuffle=True)
    x = np.array(os.listdir(train_path))

    # for train_index, val_index in kf.split(os.listdir(train_path)):
    train_index = kfold_new_indexes.training_fold0
    val_index = kfold_new_indexes.validation_fold0

    training_images = x[train_index]
    validation_images = x[val_index]

    utility.doTraining(epoch, batch_size, optimizer, my_lr, my_momentum, my_nesterov, my_decay, unlock, weights,
                       csv_path, training_images, train_path, validation_images, train_path, validation_images,
                       train_path, log_name)


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

    for i in [1, 0]:
        print("epochs: {}, bs: {}, unlock: {}, pesi: {}, opt: {}, lr: {}, mom: {}, nest: {}, dec: {}".format(
            epoch, batch_size, unlock, weights, utility.optimizers[i], utility.lrs[i], utility.moms[i],
            utility.nesterovs[i], utility.decays[i]))

        main(epoch, batch_size, unlock, weights, utility.optimizers[i], utility.lrs[i], utility.moms[i],
             utility.nesterovs[i], utility.decays[i], "new_kf_Open/Close_" + str(i))
        print("Training succesfully")
