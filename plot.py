# from matplotlib import pyplot as plt
#
#
# train_acc = [0.5383883, 0.5490958, 0.55361676, 0.57447654, 0.59287757, 0.60112625, 0.61611676, 0.6282519, 0.6380869, 0.65212566, 0.6601364, 0.6713991, 0.679172, 0.69384515, 0.7002697, 0.698842, 0.710422, 0.7190673, 0.71875, 0.7345336, 0.7367544, 0.7396891, 0.7456377, 0.75182426, 0.7612627, 0.75333124, 0.76705265, 0.7662595, 0.77434963, 0.7795051, 0.776967, 0.7851364, 0.788547, 0.7902126, 0.79600257, 0.8029822, 0.8072652, 0.81004125, 0.80663073, 0.8090895, 0.81170684, 0.8172589, 0.82074875, 0.81836927, 0.8243972, 0.82899743, 0.8248731, 0.83383566, 0.8388325, 0.82669735, 0.8282836, 0.83145624, 0.83209074, 0.842243, 0.8425603, 0.842243, 0.8466053, 0.84605014, 0.8475571, 0.851047, 0.85049176, 0.8592164, 0.8562817]
# valid_acc = [0.01171875, 0.0, 0.03515625, 0.04296875, 0.21354167, 0.0390625, 0.21354167, 0.17057292, 0.37109375, 0.44270834, 0.30208334, 0.43229166, 0.546875, 0.24348958, 0.51171875, 0.578125, 0.34375, 0.1953125, 0.2955729, 0.18098958, 0.32552084, 0.23567708, 0.38802084, 0.25, 0.54036456, 0.5, 0.2877604, 0.39583334, 0.3919271, 0.43359375, 0.37890625, 0.31770834, 0.2734375, 0.47395834, 0.47916666, 0.35416666, 0.4921875, 0.72265625, 0.4466146, 0.26822916, 0.4908854, 0.2669271, 0.4127604, 0.3046875, 0.41927084, 0.36328125, 0.34765625, 0.4361979, 0.4596354, 0.36979166, 0.5494792, 0.4752604, 0.16927083, 0.58984375, 0.34375, 0.4153646, 0.5989583, 0.46354166, 0.328125, 0.48177084, 0.37109375, 0.3424479, 0.3736979]
# train_loss = [0.6924482335899081, 0.6857549160870199, 0.6834304148170549, 0.6784020495898833, 0.6720540024302333, 0.6654969462888495, 0.6578253513665369, 0.6500870509801178, 0.6399425870270898, 0.6306907846237803, 0.6191818227017591, 0.6117115771104842, 0.6011082395381734, 0.5856039994561733, 0.5756993174250356, 0.5744615322744786, 0.5615569899227414, 0.5532322855770285, 0.5524252713634278, 0.5352316820984564, 0.5334886966017902, 0.5219431104998903, 0.5150432403620124, 0.5064892284761225, 0.4979971969490729, 0.5026676063307651, 0.48797590962521314, 0.4821685191035876, 0.4719645176744703, 0.4671887511831855, 0.4677244341615493, 0.4585895834840494, 0.44551724679578986, 0.4464290497569263, 0.437677118681409, 0.42723350204186994, 0.4211787185088027, 0.4230855792609568, 0.42355075660090763, 0.41981596026928897, 0.41072795899386333, 0.3991603753742228, 0.39837368082274033, 0.39308744752165026, 0.38782388448412647, 0.38379673190830926, 0.38838248128818376, 0.3769557499492229, 0.36647130223700236, 0.38064886047150276, 0.38145273802849244, 0.3742095400688007, 0.3714955851511302, 0.36028065127769704, 0.35465970900155563, 0.35364480581380386, 0.3486807218511698, 0.34694517392494956, 0.3514653213586904, 0.3380323610632553, 0.3392393783415635, 0.3286265442667879, 0.3364493814367933]
# valid_loss = [0.7551652689774832, 0.901385709643364, 0.7772761434316635, 0.821906233827273, 0.7981691559155782, 0.9853001485268275, 0.8544786274433136, 0.9503697156906128, 0.8024486700693766, 0.7556052654981613, 0.9481663902600607, 0.7858203401168188, 0.6990305483341217, 1.131199985742569, 0.7646582176287969, 0.6703686068455378, 1.0364802082379658, 1.415939102570216, 1.1065519899129868, 1.528528869152069, 1.121578981479009, 1.449972500403722, 1.0297325104475021, 1.4272241393725078, 0.8144641667604446, 0.855211615562439, 1.3473120431105297, 1.1105738828579585, 1.18785493572553, 1.1207559456427891, 1.2057889501253765, 1.3904064893722534, 1.8024213314056396, 1.0434679985046387, 1.1358618686596553, 1.5279548466205597, 1.092632457613945, 0.5330245643854141, 1.1839213371276855, 2.04059636592865, 1.1157941321531932, 2.0041654109954834, 1.421145071585973, 1.9344262679417927, 1.5340442260106404, 1.708501011133194, 1.7662531236807506, 1.4520746767520905, 1.3811360895633698, 1.8343599339326222, 0.968981146812439, 1.334604541460673, 3.125720043977102, 0.8888588398694992, 2.019038279851278, 1.6840244233608246, 0.8572097768386205, 1.5842724442481995, 2.23336598277092, 1.4131905436515808, 1.9305533270041149, 2.254122773806254, 2.0073502560456595]
#
# print len(train_acc)
#
# #  "Accuracy"
# plt.plot(train_acc)
# plt.plot(valid_acc)
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()
# # "Loss"
# plt.plot(train_loss)
# plt.plot(valid_loss)
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()

# from keras.optimizers import *
# optimizers = []
# lrs = [0.0001, 0.0001,
#        0.00001, 0.0001, 0.001,
#        0.0001, 0.001,
#        0.01, 0.001, 0.0001,
#        0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
#        0.001, 0.001, 0.001, 0.001, 0.001, 0.001,
#        0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001,
#        0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
#        0.01, 0.001, 0.0001,
#        0.01, 0.001, 0.0001]
# moms = [.9, .9,
#         None, None, None,
#         None, None,
#         .9, .9, .9,
#         .0, .2, .4, .6, .8, .9,
#         .0, .2, .4, .6, .8, .9,
#         .0, .2, .4, .6, .8, .9,
#         .0, .2, .4, .6, .8, .9,
#         .9, .9, .9,
#         .9, .9, .9]
# nesterovs = [False, True,
#              None, None, None,
#              None, None,
#              True, True, True,
#              False, False, False, False, False, False,
#              False, False, False, False, False, False,
#              False, False, False, False, False, False,
#              False, False, False, False, False, False,
#              True, True, True,
#              True, True, True]
# decays = [None, None,
#           None, None, None,
#           None, None,
#           1e-6, 1e-6, 1e-6,
#           None, None, None, None, None, None,
#           None, None, None, None, None, None,
#           None, None, None, None, None, None,
#           None, None, None, None, None, None,
#           1e-6, 1e-6, 1e-6,
#           1e-6, 1e-6, 1e-6]
# optimizers.append((SGD(lr=lrs[0], momentum=0.9), "SGD"))
# optimizers.append((SGD(lr=lrs[1], momentum=0.9, nesterov=True), "SGD"))
# optimizers.append((Adam(lr=lrs[2]), "Adam"))
# optimizers.append((Adam(lr=lrs[3]), "Adam"))
# optimizers.append((Adam(lr=lrs[4]), "Adam"))
# optimizers.append((RMSprop(lr=lrs[5]), "RMSprop"))
# optimizers.append((RMSprop(lr=lrs[6]), "RMSprop"))
#
# optimizers.append((SGD(lr=lrs[7], momentum=0.9, nesterov=True, decay=1e-6), "SGD"))
# optimizers.append((SGD(lr=lrs[8], momentum=0.9, nesterov=True, decay=1e-6), "SGD"))
# optimizers.append((SGD(lr=lrs[9], momentum=0.9, nesterov=True, decay=1e-6), "SGD"))
# optimizers.append((SGD(lr=lrs[10], momentum=0.0), "SGD"))
# optimizers.append((SGD(lr=lrs[11], momentum=0.2), "SGD"))
# optimizers.append((SGD(lr=lrs[12], momentum=0.4), "SGD"))
# optimizers.append((SGD(lr=lrs[13], momentum=0.6), "SGD"))
# optimizers.append((SGD(lr=lrs[14], momentum=0.8), "SGD"))
# optimizers.append((SGD(lr=lrs[15], momentum=0.9), "SGD"))
#
# optimizers.append((SGD(lr=lrs[16], momentum=0.0), "SGD"))
# optimizers.append((SGD(lr=lrs[17], momentum=0.2), "SGD"))
# optimizers.append((SGD(lr=lrs[18], momentum=0.4), "SGD"))
# optimizers.append((SGD(lr=lrs[19], momentum=0.6), "SGD"))
# optimizers.append((SGD(lr=lrs[20], momentum=0.8), "SGD"))
# optimizers.append((SGD(lr=lrs[21], momentum=0.9), "SGD"))
#
# optimizers.append((SGD(lr=lrs[22], momentum=0.0), "SGD"))
# optimizers.append((SGD(lr=lrs[23], momentum=0.2), "SGD"))
# optimizers.append((SGD(lr=lrs[24], momentum=0.4), "SGD"))
# optimizers.append((SGD(lr=lrs[25], momentum=0.6), "SGD"))
# optimizers.append((SGD(lr=lrs[26], momentum=0.8), "SGD"))
# optimizers.append((SGD(lr=lrs[27], momentum=0.9), "SGD"))
#
# optimizers.append((SGD(lr=lrs[28], momentum=0.0), "SGD"))
# optimizers.append((SGD(lr=lrs[29], momentum=0.2), "SGD"))
# optimizers.append((SGD(lr=lrs[30], momentum=0.4), "SGD"))
# optimizers.append((SGD(lr=lrs[31], momentum=0.6), "SGD"))
# optimizers.append((SGD(lr=lrs[32], momentum=0.8), "SGD"))
# optimizers.append((SGD(lr=lrs[33], momentum=0.9), "SGD"))
#
# optimizers.append((SGD(lr=lrs[34], momentum=0.9, nesterov=True, decay=5e-5), "SGD"))
# optimizers.append((SGD(lr=lrs[35], momentum=0.9, nesterov=True, decay=5e-5), "SGD"))
# optimizers.append((SGD(lr=lrs[36], momentum=0.9, nesterov=True, decay=5e-5), "SGD"))
#
# optimizers.append((SGD(lr=lrs[37], momentum=0.9, nesterov=True, decay=1e-5), "SGD"))
# optimizers.append((SGD(lr=lrs[38], momentum=0.9, nesterov=True, decay=1e-5), "SGD"))
# optimizers.append((SGD(lr=lrs[39], momentum=0.9, nesterov=True, decay=1e-5), "SGD"))
#
#
# for index, elem in enumerate(optimizers):
#     print(elem[1], "lrs:", lrs[index], "moms:", moms[index], "nesterovs:", nesterovs[index], "decays:", decays[index])


# import os
# import csv
#
# valCsvPath = "/Users/alex/Desktop/bone age/Validation Dataset.csv"
# valPath = '/Users/alex/Desktop/bone age/validation/boneage-validation-dataset-1/'
# testPath = '/Users/alex/Desktop/bone age/validation/boneage-validation-dataset-2/'
#
# csvDict = {}
#
# path = testPath
# reader = csv.reader(open(valCsvPath, 'r'))
# for row in reader:
#     if row[1] != "male":
#         csvDict[row[0]] = row[1]
#
# n1 = os.listdir(path)
#
# male = 0
# female = 0
# n = len(n1)
#
# print("\n\n")
# for i in range(n):
#     # print(n1[i])
#
#     expected = csvDict[n1[i].split(".")[0]]
#
#     # print(prediction, expected)
#
#     if expected.lower() == "true":
#         male += 1
#     elif expected.lower() == "false":
#         female += 1
#
# print(n, male, female)

# return model.predict(train_img, verbose=1)
# print(csvDict[n1[i].split(".")[0]])


# import cv2
# import os
#
# masksTrainPath = '/data/segmentation2/train_masks/train/'
# masksvalPath = '/data/segmentation2/val_masks/val/'
# masksTestPath = '/data/segmentation2/test_masks/test/'
#
# trainPath = '/data/reduced_masked_normalized/training/'
# valPath = '/data/reduced_masked_normalized/validation1/'
# testPath = '/data/reduced_masked_normalized/validation2/'
#
#
# def get_shape_factor(folder):
#     imgs = os.listdir(folder)
#     # shape_factor = 0
#     max_sf = 0
#     min_sf = 500
#     for img in imgs:
#         image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
#         image, contours, hierarchy = cv2.findContours(image, 1, 2)
#         cnt = contours[0]
#         area = cv2.contourArea(cnt)
#         perimeter = cv2.arcLength(cnt, True)
#
#         sf = area / perimeter / perimeter
#         if sf > max_sf:
#             max_sf = sf
#         elif sf < min_sf:
#             min_sf = sf
#
#         # shape_factor += sf
#     return max_sf, min_sf
#
#
# def filter_by_shape_form(folder, min_sf, max_sf):
#     imgs = os.listdir(folder)
#     probably_wrong = []
#     for img in imgs:
#         image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
#         image, contours, hierarchy = cv2.findContours(image, 1, 2)
#         cnt = contours[0]
#         area = cv2.contourArea(cnt)
#         perimeter = cv2.arcLength(cnt, True)
#
#         sf = area / perimeter / perimeter
#
#         if min_sf < sf < max_sf:
#             probably_wrong.append(img)
#             os.remove(img)
#
#     print(probably_wrong)
#
#
# removed_test1 = ['10461.png', '14387.png', '12242.png', '13308.png', '11228.png']
# removed_val1 = ['3117.png', '6231.png', '3625.png', '2426.png', '1532.png', '2981.png', '1950.png', '1795.png', '1397.png', '9536.png', '1676.png', '5865.png', '9559.png', '4800.png']
# removed_train1 = ['4284.png', '4677.png', '4128.png', '1761.png', '14399.png', '4193.png', '6235.png', '14992.png', '4230.png', '3633.png', '3503.png', '3918.png', '3823.png', '7480.png', '2414.png', '8757.png', '1668.png', '8180.png', '11079.png', '12993.png', '3395.png', '13674.png', '14492.png', '1702.png', '3437.png', '4072.png', '4217.png', '5118.png', '1840.png', '7122.png', '7907.png', '1418.png', '3883.png', '9403.png', '1826.png', '13088.png', '7458.png', '15312.png', '6985.png', '7507.png', '5035.png', '4270.png', '15295.png', '3287.png', '1609.png', '3964.png', '14708.png', '2563.png', '9077.png', '11829.png', '9426.png', '11863.png', '10413.png', '9846.png', '12201.png', '9299.png', '10424.png', '6044.png', '15368.png', '3145.png', '1378.png', '6961.png', '1458.png', '2862.png', '9165.png', '12167.png', '3931.png', '6273.png', '14281.png', '2146.png', '11152.png', '2058.png', '6886.png', '3999.png', '13888.png', '2482.png', '7493.png', '9759.png', '10820.png', '15276.png', '10895.png', '12761.png', '13404.png', '1541.png', '12832.png', '3389.png', '13273.png', '4069.png', '6096.png', '14621.png', '12036.png', '7680.png', '1618.png', '3515.png', '6937.png', '10042.png', '1779.png', '13021.png', '11404.png', '2193.png', '8668.png', '8580.png', '1473.png', '2194.png', '3157.png', '1446.png', '1398.png', '8940.png', '14179.png', '3885.png', '3653.png', '7794.png', '4004.png', '6152.png', '10627.png', '6319.png', '8100.png', '13590.png', '7990.png', '1786.png', '9775.png', '3899.png', '1799.png', '15462.png', '5826.png', '9031.png', '1698.png', '8710.png', '11713.png', '3977.png', '6801.png', '1711.png', '11042.png', '2881.png', '6264.png', '10390.png', '3827.png', '1592.png', '5711.png', '8963.png', '1905.png', '4905.png', '1912.png', '3183.png', '8607.png']
#
# removed_test2 = ['11156.png', '11438.png', '10461.png', '12510.png', '10543.png', '14387.png', '12147.png', '12792.png', '11146.png', '12242.png', '9755.png', '14111.png', '11515.png', '13308.png', '12963.png', '14792.png', '10389.png', '11692.png', '11228.png', '13345.png', '12909.png']
# removed_val2 = ['2016.png', '3733.png', '3117.png', '3102.png', '1392.png', '6231.png', '6165.png', '3625.png', '5033.png', '3699.png', '2199.png', '6211.png', '3404.png', '3869.png', '2426.png', '1450.png', '2910.png', '1603.png', '7705.png', '1633.png', '2675.png', '6891.png', '2335.png', '1532.png', '3795.png', '7777.png', '9674.png', '4095.png', '3143.png', '3932.png', '4338.png', '2981.png', '1950.png', '1795.png', '8072.png', '1397.png', '9536.png', '4156.png', '5685.png', '3727.png', '1908.png', '3221.png', '2549.png', '3326.png', '2842.png', '1788.png', '3422.png', '1735.png', '2338.png', '3107.png', '3009.png', '2869.png', '1693.png', '2036.png', '1676.png', '4274.png', '5865.png', '4127.png', '2047.png', '9007.png', '3442.png', '9559.png', '3762.png', '1861.png', '4800.png', '6649.png', '2198.png', '2569.png', '3466.png', '2397.png', '3109.png', '1655.png', '3106.png', '3322.png', '2863.png', '5099.png']
# removed_train2 = ['2496.png', '4284.png', '4089.png', '10836.png', '6035.png', '4677.png', '5270.png', '7738.png', '4128.png', '4978.png', '1761.png', '3992.png', '3023.png', '6433.png', '3832.png', '14399.png', '9516.png', '3728.png', '11336.png', '4611.png', '8877.png', '3028.png', '11762.png', '7869.png', '3217.png', '1825.png', '3034.png', '4193.png', '1705.png', '1422.png', '3592.png', '5844.png', '6457.png', '4669.png', '1632.png', '6403.png', '11022.png', '3849.png', '2697.png', '12514.png', '2964.png', '6235.png', '6432.png', '14992.png', '7235.png', '4230.png', '3539.png', '3633.png', '3503.png', '3918.png', '10431.png', '3681.png', '3021.png', '3823.png', '7480.png', '11057.png', '3926.png', '8235.png', '11375.png', '3541.png', '3411.png', '4266.png', '6055.png', '3405.png', '5645.png', '3501.png', '4021.png', '3586.png', '3029.png', '8735.png', '5512.png', '11872.png', '14963.png', '10727.png', '3542.png', '2414.png', '4291.png', '2814.png', '8757.png', '6038.png', '12727.png', '1844.png', '1668.png', '2400.png', '14557.png', '5927.png', '8295.png', '14437.png', '2241.png', '6154.png', '1831.png', '6407.png', '3408.png', '2122.png', '8180.png', '3639.png', '11195.png', '2660.png', '3750.png', '3206.png', '2718.png', '1429.png', '3155.png', '11706.png', '2389.png', '11416.png', '13018.png', '9721.png', '2436.png', '2976.png', '3673.png', '4610.png', '3384.png', '11079.png', '1494.png', '12993.png', '9269.png', '3548.png', '5319.png', '4216.png', '2899.png', '3798.png', '3693.png', '11377.png', '15378.png', '10059.png', '13675.png', '3395.png', '2249.png', '13674.png', '14492.png', '1702.png', '3437.png', '13154.png', '4218.png', '11604.png', '1737.png', '3414.png', '2292.png', '3773.png', '6006.png', '6576.png', '13410.png', '2901.png', '4323.png', '4072.png', '4217.png', '13123.png', '11465.png', '7249.png', '1694.png', '3615.png', '5410.png', '3771.png', '5118.png', '3905.png', '5791.png', '3783.png', '2573.png', '4249.png', '8182.png', '14706.png', '3186.png', '2621.png', '3260.png', '10247.png', '14041.png', '3558.png', '3547.png', '1840.png', '7122.png', '14996.png', '4290.png', '12282.png', '3843.png', '3071.png', '7907.png', '4131.png', '1418.png', '13532.png', '2383.png', '12548.png', '3209.png', '3883.png', '2359.png', '2845.png', '3135.png', '14742.png', '7812.png', '13563.png', '10384.png', '10020.png', '2108.png', '2375.png', '3848.png', '10701.png', '11460.png', '9403.png', '12680.png', '1826.png', '1938.png', '13088.png', '4147.png', '9105.png', '8083.png', '6174.png', '12912.png', '4687.png', '7458.png', '2724.png', '3332.png', '13664.png', '4053.png', '15312.png', '6985.png', '1684.png', '4276.png', '7507.png', '5035.png', '1823.png', '3002.png', '15003.png', '4270.png', '4238.png', '12452.png', '3754.png', '15593.png', '15295.png', '2606.png', '3287.png', '3179.png', '1609.png', '3964.png', '5426.png', '2903.png', '2545.png', '2825.png', '11211.png', '4159.png', '2106.png', '14708.png', '3223.png', '6421.png', '14822.png', '2707.png', '1935.png', '1746.png', '2926.png', '2563.png', '9737.png', '2701.png', '2810.png', '3194.png', '7042.png', '14837.png', '2099.png', '3597.png', '12657.png', '4347.png', '2609.png', '12558.png', '9077.png', '11829.png', '10915.png', '3073.png', '3255.png', '5735.png', '3141.png', '3523.png', '4334.png', '7240.png', '11620.png', '15512.png', '9426.png', '2366.png', '2207.png', '13086.png', '4052.png', '3195.png', '11863.png', '13626.png', '10413.png', '14343.png', '12756.png', '6317.png', '8870.png', '7666.png', '9712.png', '1739.png', '9823.png', '2156.png', '7784.png', '9846.png', '8752.png', '5070.png', '1405.png', '14639.png', '1738.png', '13193.png', '12201.png', '11488.png', '7304.png', '3497.png', '4067.png', '1920.png', '15064.png', '11313.png', '5568.png', '3922.png', '3375.png', '3197.png', '9299.png', '1784.png', '2787.png', '11501.png', '2010.png', '2123.png', '3020.png', '9341.png', '10424.png', '10697.png', '15555.png', '2808.png', '9988.png', '9233.png', '3715.png', '5213.png', '6044.png', '1411.png', '14804.png', '7232.png', '2595.png', '4100.png', '2261.png', '1538.png', '3980.png', '4038.png', '15368.png', '1897.png', '3145.png', '5366.png', '1378.png', '11963.png', '4921.png', '2030.png', '9620.png', '1802.png', '14152.png', '10121.png', '13458.png', '11491.png', '7735.png', '2505.png', '14620.png', '13824.png', '1851.png', '6961.png', '1399.png', '12037.png', '6999.png', '2283.png', '12190.png', '1771.png', '1458.png', '1701.png', '2862.png', '7381.png', '9584.png', '7637.png', '11880.png', '4799.png', '9165.png', '10955.png', '3019.png', '10004.png', '12167.png', '6032.png', '3931.png', '3054.png', '13797.png', '4342.png', '9634.png', '1998.png', '2044.png', '14853.png', '13358.png', '6499.png', '6273.png', '3337.png', '14281.png', '3232.png', '3321.png', '2146.png', '3772.png', '1597.png', '3796.png', '1545.png', '14798.png', '8783.png', '7636.png', '4036.png', '11152.png', '11453.png', '2879.png', '8074.png', '3325.png', '3115.png', '2113.png', '5415.png', '1521.png', '2058.png', '6886.png', '15243.png', '6794.png', '2479.png', '3985.png', '3590.png', '3999.png', '2907.png', '5704.png', '10316.png', '13888.png', '11046.png', '3219.png', '3623.png', '2605.png', '7761.png', '3348.png', '4305.png', '2042.png', '3530.png', '15282.png', '6312.png', '3504.png', '9004.png', '3096.png', '3436.png', '2134.png', '2482.png', '3669.png', '3656.png', '8449.png', '10663.png', '2731.png', '4099.png', '7493.png', '2584.png', '10645.png', '12744.png', '2102.png', '12713.png', '5831.png', '4355.png', '9178.png', '9759.png', '10820.png', '1590.png', '5970.png', '15276.png', '3877.png', '7963.png', '10895.png', '3577.png', '2001.png', '14836.png', '1457.png', '12761.png', '12601.png', '13404.png', '10242.png', '7349.png', '3545.png', '5238.png', '11949.png', '3341.png', '12873.png', '7017.png', '2262.png', '3806.png', '1541.png', '12832.png', '8218.png', '2060.png', '2037.png', '6385.png', '3389.png', '2202.png', '2276.png', '7057.png', '13273.png', '2320.png', '2955.png', '12108.png', '3291.png', '5795.png', '2908.png', '3406.png', '13693.png', '3589.png', '14044.png', '10806.png', '4001.png', '15120.png', '2837.png', '5794.png', '4281.png', '4069.png', '8102.png', '2411.png', '2559.png', '3865.png', '5335.png', '6062.png', '2866.png', '14584.png', '11439.png', '6096.png', '7651.png', '2878.png', '3076.png', '3631.png', '13135.png', '13638.png', '15314.png', '7671.png', '11327.png', '4818.png', '4304.png', '3434.png', '10107.png', '14621.png', '2278.png', '2256.png', '2704.png', '2669.png', '12036.png', '7680.png', '1618.png', '3515.png', '4073.png', '3094.png', '2661.png', '6937.png', '10042.png', '1779.png', '13021.png', '11404.png', '3981.png', '2193.png', '8668.png', '12035.png', '3751.png', '15363.png', '8580.png', '3090.png', '3196.png', '4575.png', '2385.png', '3056.png', '14466.png', '13343.png', '5041.png', '3598.png', '4076.png', '8884.png', '14882.png', '1473.png', '4244.png', '3700.png', '9653.png', '2194.png', '3176.png', '3032.png', '3157.png', '6405.png', '2543.png', '4300.png', '14710.png', '3708.png', '2303.png', '2214.png', '3159.png', '2089.png', '3387.png', '1446.png', '14681.png', '1398.png', '8940.png', '3792.png', '14179.png', '3394.png', '2962.png', '3955.png', '7416.png', '15588.png', '8566.png', '2691.png', '12996.png', '3885.png', '3653.png', '8404.png', '2039.png', '10768.png', '7794.png', '8868.png', '4004.png', '1530.png', '6152.png', '3867.png', '10627.png', '3415.png', '6319.png', '1725.png', '3596.png', '3935.png', '6607.png', '4328.png', '3937.png', '3178.png', '3465.png', '3464.png', '2829.png', '2883.png', '4307.png', '2626.png', '11114.png', '4329.png', '8100.png', '3712.png', '14037.png', '3913.png', '2248.png', '5589.png', '13590.png', '6936.png', '3215.png', '7990.png', '9995.png', '9447.png', '1786.png', '3376.png', '10146.png', '2716.png', '3864.png', '4243.png', '9775.png', '12683.png', '3899.png', '3559.png', '2959.png', '12978.png', '4028.png', '1799.png', '5169.png', '2513.png', '15462.png', '5826.png', '9533.png', '3511.png', '1660.png', '4339.png', '11458.png', '9031.png', '2258.png', '11645.png', '6296.png', '11616.png', '4071.png', '4163.png', '2629.png', '13054.png', '9953.png', '1698.png', '8710.png', '11713.png', '5096.png', '4792.png', '2309.png', '5399.png', '3977.png', '2848.png', '9649.png', '15294.png', '5309.png', '11897.png', '2966.png', '6801.png', '2023.png', '1617.png', '2388.png', '15437.png', '3253.png', '4105.png', '1711.png', '3396.png', '7140.png', '2208.png', '9259.png', '3688.png', '14229.png', '11042.png', '7656.png', '3670.png', '11447.png', '11268.png', '4627.png', '2091.png', '5251.png', '2881.png', '7828.png', '13379.png', '3127.png', '1868.png', '10769.png', '9122.png', '6264.png', '11467.png', '10390.png', '7007.png', '3827.png', '1592.png', '2324.png', '12502.png', '2911.png', '2305.png', '5711.png', '3998.png', '9966.png', '8963.png', '8617.png', '1905.png', '6654.png', '6692.png', '15604.png', '3014.png', '2021.png', '14126.png', '2293.png', '1646.png', '2407.png', '4905.png', '3420.png', '1542.png', '2667.png', '2993.png', '7217.png', '5358.png', '14806.png', '9715.png', '3722.png', '3986.png', '1889.png', '2456.png', '6863.png', '5698.png', '4959.png', '4096.png', '6224.png', '2321.png', '11236.png', '2251.png', '2492.png', '4187.png', '1912.png', '3522.png', '3973.png', '2965.png', '2350.png', '3534.png', '8992.png', '3183.png', '3780.png', '8607.png', '5834.png', '14370.png', '8706.png']
#
# # print(len(removed_test1))
# # print(len(removed_val1))
# # print(len(removed_train1))
# # print(len(removed_test2))
# # print(len(removed_val2))
# # print(len(removed_train2))
#
# #test = ['14387.png', '13308.png']
# #val = ['3625.png', '1392.png', '1397.png', '9536.png', '5865.png', '2016.png', '1450.png', '3326.png', '5033.png', '4338.png', '1693.png', '2426.png', '2036.png', '3117.png', '1676.png', '9559.png', '1908.png', '9007.png', '4800.png', '3404.png', '1603.png', '6231.png', '5099.png', '2047.png']
# # train = ['13358.png', '8607.png', '4217.png', '14621.png', '3395.png', '5041.png', '7656.png', '1541.png', '8940.png', '3183.png', '6038.png', '3633.png', '2505.png', '15462.png', '7680.png', '3223.png', '3023.png', '3590.png', '2881.png', '3547.png', '8566.png', '7480.png', '4610.png', '3631.png', '3157.png', '2249.png', '6985.png', '9447.png', '5645.png', '14179.png', '2102.png', '7907.png', '5309.png', '1609.png', '3977.png', '1802.png', '3998.png', '13404.png', '14343.png', '5319.png', '3387.png', '3436.png', '7122.png', '10247.png', '5251.png', '12108.png', '2258.png', '1779.png', '3806.png', '3437.png', '11616.png', '2661.png', '3219.png', '3999.png', '10697.png', '3964.png', '1632.png', '4004.png', '3596.png', '5118.png', '1418.png', '6405.png', '3145.png', '1746.png', '4818.png', '7493.png', '1784.png', '3434.png', '2058.png', '4575.png', '3597.png', '1668.png', '5426.png', '10424.png', '9988.png', '9988.png', '11706.png', '4300.png', '13273.png', '8710.png', '9341.png', '6264.png', '1457.png', '4249.png', '11713.png', '1494.png', '12756.png', '10627.png', '9584.png', '1694.png', '6886.png', '13674.png', '11467.png', '4128.png', '3135.png', '3653.png', '1905.png', '3337.png', '11880.png', '6273.png', '5791.png', '2808.png', '8100.png', '4921.png', '1646.png', '12037.png', '6317.png', '7828.png', '13088.png', '1530.png', '3864.png', '15604.png', '4677.png', '3656.png', '2010.png', '4307.png', '7140.png', '5970.png', '9031.png', '8449.png', '2414.png', '12036.png', '3722.png', '3905.png', '14492.png', '1889.png', '4067.png', '4073.png', '9966.png', '10390.png', '14041.png', '6937.png', '3669.png', '2563.png', '3899.png', '11460.png', '6936.png', '9620.png', '4304.png', '4270.png', '1446.png', '2359.png', '7235.png', '10820.png', '9759.png', '15555.png', '3260.png', '7416.png', '2023.png', '13086.png', '6319.png', '7761.png', '6457.png', '12993.png', '1660.png', '8877.png', '12978.png', '1851.png', '14853.png', '11042.png', '3511.png', '8083.png', '12167.png', '3883.png', '3115.png', '3464.png', '3504.png', '10413.png', '2626.png', '4069.png', '1868.png', '1840.png', '3931.png', '1897.png', '4284.png', '2193.png', '9165.png', '13532.png', '4131.png', '1935.png', '6096.png', '1538.png', '15368.png', '2964.png', '12201.png', '3501.png', '5831.png', '4905.png', '3918.png', '6692.png', '4052.png', '15593.png', '3503.png', '4244.png', '14706.png', '5589.png', '3885.png', '1738.png', '1739.png', '9403.png', '1705.png', '13054.png', '11079.png', '9775.png', '2146.png', '5826.png', '3700.png', '1831.png', '3773.png', '2479.png', '2848.png', '11963.png', '9712.png', '4342.png', '6794.png', '13797.png', '3389.png', '1429.png', '6385.png', '1617.png', '3955.png', '1826.png', '8180.png', '6152.png', '9077.png', '12514.png', '5711.png', '3823.png', '8963.png', '14992.png', '11829.png', '10316.png', '11949.png', '11236.png', '14399.png', '1823.png', '15295.png', '3577.png', '4230.png', '13021.png', '6235.png', '1771.png', '3827.png', '9649.png', '9846.png', '7738.png', '15243.png', '3589.png', '14822.png', '5844.png', '1405.png', '1411.png', '3548.png', '3548.png', '9105.png', '4347.png', '8757.png', '7458.png', '1799.png', '4193.png', '13590.png', '11152.png', '9299.png']
#
#
# print(len(removed_train1))

import os
import cv2

### TEST
# bad_segmented = ['14807.png', '15448.png', '14457.png', '11632.png']
# to_remove = ['11312.png', '11183.png']
# to_rotate = ['12494.png', '13417.png', '15448.png', '15076.png', '14325.png', '9743.png', '12814.png', '10601.png', '12110.png', '12868.png', '14767.png', '10649.png']
# to_flip = []
test_to_remove = ['14807.png', '15448.png', '14457.png', '11632.png', '11312.png', '11183.png', '12494.png',
                  '13417.png', '15448.png', '15076.png', '14325.png', '9743.png', '12814.png', '10601.png', '12110.png',
                  '12868.png', '14767.png', '10649.png']

## VALIDATION
# bad_segmented =  ['1421.png', '1583.png', '9240.png', '1769.png', '3769.png', '3965.png', '3862.png', '1537.png', '2820.png', '7629.png', '3498.png', '4733.png', '4051.png', '2434.png', '4125.png', '3139.png', '2972.png', '3852.png', '2257.png', '1774.png', '2335.png', '4022.png', '5847.png', '1611.png', '1822.png', '8804.png', '4025.png', '8352.png', '7075.png', '3944.png']
# to_remove =  ['7790.png', '2945.png', '3868.png', '9607.png', '4119.png', '2190.png']
# to_rotate =  ['2302.png', '2459.png', '1386.png', '1579.png', '4274.png', '3452.png', '5802.png', '7196.png', '9580.png', '3683.png', '6507.png', '8853.png', '9361.png', '7371.png', '9638.png', '8646.png', '6496.png', '2140.png', '5388.png', '6347.png', '1788.png', '2054.png', '9673.png', '1401.png', '5883.png', '4009.png', '7853.png', '7476.png', '4022.png', '4783.png', '1836.png', '5702.png', '8352.png', '3575.png']
# to_flip =  []
validation_to_remove = ['1421.png', '1583.png', '9240.png', '1769.png', '3769.png', '3965.png', '3862.png', '1537.png',
                        '2820.png', '7629.png', '3498.png', '4733.png', '4051.png', '2434.png', '4125.png', '3139.png',
                        '2972.png', '3852.png', '2257.png', '1774.png', '2335.png', '4022.png', '5847.png', '1611.png',
                        '1822.png', '8804.png', '4025.png', '8352.png', '7075.png', '3944.png', '7790.png', '2945.png',
                        '3868.png', '9607.png', '4119.png', '2190.png', '2302.png', '2459.png', '1386.png', '1579.png',
                        '4274.png', '3452.png', '5802.png', '7196.png', '9580.png', '3683.png', '6507.png', '8853.png',
                        '9361.png', '7371.png', '9638.png', '8646.png', '6496.png', '2140.png', '5388.png', '6347.png',
                        '1788.png', '2054.png', '9673.png', '1401.png', '5883.png', '4009.png', '7853.png', '7476.png',
                        '4022.png', '4783.png', '1836.png', '5702.png', '8352.png', '3575.png']

## TRAINING
training_to_remove = ['1415.png', '6172.png', '3156.png', '4203.png', '3544.png', '2909.png', '1596.png', '1582.png',
                      '2060.png', '2706.png', '11605.png', '3430.png', '10043.png', '12898.png', '5518.png', '4174.png',
                      '12866.png', '1396.png', '13013.png', '5054.png', '2061.png', '3551.png', '8438.png', '10491.png',
                      '1636.png', '3021.png', '2471.png', '6824.png', '3976.png', '1387.png', '8599.png', '5917.png',
                      '7696.png', '8205.png', '3369.png', '2705.png', '3433.png', '15312.png', '3816.png', '1973.png',
                      '8980.png', '4163.png', '3181.png', '8573.png', '3354.png', '11613.png', '3432.png', '2062.png',
                      '9862.png', '4229.png', '9057.png', '11983.png', '3585.png', '10653.png', '14353.png', '4017.png',
                      '4942.png', '3797.png', '1625.png', '4007.png', '5682.png', '3595.png', '5127.png', '3556.png',
                      '10441.png', '2066.png', '3378.png', '7491.png', '3813.png', '3621.png', '3635.png', '7453.png',
                      '3812.png', '4199.png', '4204.png', '2107.png', '4576.png', '1426.png', '3225.png', '1432.png',
                      '1624.png', '10118.png', '1803.png', '3958.png', '3970.png', '7928.png', '14817.png', '2139.png',
                      '3569.png', '4206.png', '3848.png', '7796.png', '7184.png', '1593.png', '2059.png', '13017.png',
                      '14022.png', '6203.png', '2065.png', '2071.png', '12447.png', '3637.png', '10720.png',
                      '14546.png', '4164.png', '14552.png', '14235.png', '3811.png', '3187.png', '6558.png', '1431.png',
                      '11328.png', '3583.png', '9535.png', '2312.png', '3756.png', '1814.png', '12917.png', '9591.png',
                      '1873.png', '8102.png', '1640.png', '3731.png', '1654.png', '1481.png', '14093.png', '5793.png',
                      '2765.png', '1522.png', '3692.png', '3679.png', '3650.png', '2572.png', '10035.png', '5584.png',
                      '10948.png', '3863.png', '13702.png', '11667.png', '3491.png', '3485.png', '13306.png',
                      '1899.png', '3730.png', '3917.png', '1870.png', '13489.png', '8840.png', '1864.png', '10816.png',
                      '9545.png', '3040.png', '13476.png', '13304.png', '4288.png', '3518.png', '1469.png', '3524.png',
                      '1455.png', '9182.png', '3487.png', '4303.png', '2014.png', '2000.png', '3478.png', '5747.png',
                      '3444.png', '2982.png', '3875.png', '1910.png', '3685.png', '12813.png', '2216.png', '2558.png',
                      '7421.png', '1736.png', '1722.png', '12153.png', '2571.png', '14287.png', '1911.png', '1534.png',
                      '2983.png', '2029.png', '12351.png', '6515.png', '3486.png', '3492.png', '2175.png', '3294.png',
                      '4289.png', '1656.png', '2363.png', '2405.png', '3928.png', '8699.png', '13315.png', '3051.png',
                      '5346.png', '12023.png', '11099.png', '5150.png', '2165.png', '9742.png', '3509.png', '1478.png',
                      '7155.png', '1518.png', '9391.png', '3657.png', '4138.png', '2213.png', '9352.png', '5568.png',
                      '10026.png', '11338.png', '3124.png', '3871.png', '1928.png', '3695.png', '6504.png', '10219.png',
                      '1531.png', '15407.png', '10594.png', '11852.png', '1451.png', '9794.png', '4298.png', '6666.png',
                      '2428.png', '8844.png', '1874.png', '10186.png', '15015.png', '11273.png', '1679.png', '2364.png',
                      '2827.png', '6466.png', '3495.png', '3324.png', '7624.png', '12418.png', '1527.png', '4305.png',
                      '2012.png', '7426.png', '3898.png', '7340.png', '14525.png', '2204.png', '2210.png', '7369.png',
                      '1718.png', '11474.png', '1724.png', '3655.png', '6088.png', '2588.png', '3682.png', '2013.png',
                      '10540.png', '3457.png', '2985.png', '3494.png', '7779.png', '9026.png', '2173.png', '2365.png',
                      '8890.png', '10636.png', '1650.png', '3053.png', '4043.png', '15019.png', '2340.png', '3076.png',
                      '7576.png', '1675.png', '11041.png', '2022.png', '2585.png', '1729.png', '7358.png', '6720.png',
                      '3880.png', '1701.png', '1700.png', '1502.png', '3473.png', '1516.png', '10558.png', '7007.png',
                      '1476.png', '2143.png', '3249.png', '14689.png', '1489.png', '12987.png', '3711.png', '2341.png',
                      '1648.png', '1884.png', '3921.png', '4068.png', '1662.png', '4295.png', '8493.png', '4256.png',
                      '10214.png', '15378.png', '12417.png', '8901.png', '4875.png', '3897.png', '3100.png', '1717.png',
                      '3672.png', '2222.png', '12600.png', '4121.png', '3896.png', '6905.png', '1918.png', '11137.png',
                      '13708.png', '6534.png', '2034.png', '11057.png', '11527.png', '14463.png', '1844.png',
                      '3919.png', '8680.png', '8858.png', '3925.png', '6691.png', '11537.png', '2352.png', '12002.png',
                      '2178.png', '3514.png', '3528.png', '9159.png', '1511.png', '14711.png', '4327.png', '3448.png',
                      '4333.png', '3674.png', '7389.png', '8910.png', '3110.png', '1713.png', '3104.png', '2232.png',
                      '2226.png', '3886.png', '6041.png', '2541.png', '3688.png', '15181.png', '3449.png', '4326.png',
                      '3475.png', '7029.png', '1458.png', '2623.png', '4285.png', '2192.png', '10628.png', '14328.png',
                      '1672.png', '13484.png', '4085.png', '3715.png', '1664.png', '1880.png', '13690.png', '4293.png',
                      '12559.png', '2153.png', '12217.png', '3477.png', '2027.png', '2755.png', '10574.png', '1923.png',
                      '6094.png', '8907.png', '5210.png', '3675.png', '3649.png', '1922.png', '6917.png', '2754.png',
                      '3338.png', '3476.png', '3462.png', '12570.png', '7994.png', '4279.png', '14868.png', '4286.png',
                      '3728.png', '1659.png', '4090.png', '6877.png', '9289.png', '2337.png', '3983.png', '3767.png',
                      '13184.png', '6353.png', '11988.png', '2096.png', '2733.png', '3363.png', '3820.png', '4154.png',
                      '4168.png', '5505.png', '3174.png', '1763.png', '3606.png', '2518.png', '1987.png', '4627.png',
                      '4141.png', '11382.png', '3821.png', '9883.png', '11023.png', '3564.png', '8431.png', '4237.png',
                      '8419.png', '2861.png', '3766.png', '3969.png', '1830.png', '11579.png', '3764.png', '1601.png',
                      '15051.png', '2650.png', '2136.png', '2056.png', '1563.png', '3837.png', '13540.png', '5513.png',
                      '3611.png', '1748.png', '5248.png', '10074.png', '14212.png', '8977.png', '6796.png', '3836.png',
                      '4354.png', '14010.png', '15497.png', '3573.png', '8368.png', '1600.png', '8142.png', '3771.png',
                      '1614.png', '3759.png', '3956.png', '3007.png', '3013.png', '6382.png', '9072.png', '2133.png',
                      '2655.png', '13784.png', '6209.png', '10500.png', '1764.png', '8740.png', '5516.png', '1995.png',
                      '3628.png', '5259.png', '1956.png', '3833.png', '13593.png', '5729.png', '6546.png', '5073.png',
                      '2085.png', '5305.png', '13424.png', '1605.png', '3012.png', '10852.png', '8192.png', '3951.png',
                      '1607.png', '13426.png', '3004.png', '6632.png', '1388.png', '7921.png', '3574.png', '3212.png',
                      '4569.png', '13156.png', '2130.png', '5893.png', '2642.png', '2939.png', '2087.png', '2078.png',
                      '1559.png', '1798.png', '3616.png', '4179.png', '4623.png', '10927.png', '12301.png', '13989.png',
                      '2086.png', '7049.png', '11740.png', '3207.png', '1404.png', '2327.png', '5448.png', '9266.png',
                      '8150.png', '4018.png', '1835.png', '3978.png', '7871.png', '1804.png', '3961.png', '3949.png',
                      '3791.png', '1838.png', '3785.png', '9525.png', '2316.png', '6164.png', '10645.png', '4773.png',
                      '6602.png', '3752.png', '1637.png', '5860.png', '7911.png', '10484.png', '7093.png', '2882.png',
                      '11765.png', '12495.png', '13947.png', '7118.png', '9109.png', '1555.png', '10902.png',
                      '1958.png', '12867.png', '6762.png', '4613.png', '2262.png', '9323.png', '3168.png', '4161.png',
                      '7468.png', '8983.png', '8773.png', '1743.png', '8001.png', '7454.png', '11375.png', '3632.png',
                      '5242.png', '8772.png', '13211.png', '1756.png', '8982.png', '4160.png', '2288.png', '1781.png',
                      '1959.png', '6788.png', '3828.png', '8969.png', '1965.png', '1971.png', '11176.png', '3431.png',
                      '5726.png', '2049.png', '1554.png', '11604.png', '11162.png', '6213.png', '12325.png', '1568.png',
                      '6561.png', '2707.png', '11837.png', '2883.png', '1434.png', '2129.png', '2667.png', '15258.png',
                      '12537.png', '14620.png', '6415.png', '12523.png', '1385.png', '3592.png', '2868.png', '8606.png',
                      '8148.png', '4000.png', '7290.png', '10875.png', '3779.png', '12723.png', '3989.png', '7537.png',
                      '15070.png', '11969.png', '1378.png', '1422.png', '14781.png', '1581.png', '4166.png', '2077.png',
                      '12327.png', '12441.png', '3341.png', '1556.png', '10083.png', '1783.png', '4189.png', '2513.png',
                      '4162.png', '14232.png', '6013.png', '1768.png', '9334.png', '4176.png', '5240.png', '1998.png',
                      '15104.png', '7324.png', '3630.png', '12642.png', '10055.png', '9321.png', '1782.png', '1972.png',
                      '1966.png', '5057.png', '2076.png', '9876.png', '2923.png', '1594.png', '2937.png', '1437.png',
                      '3591.png', '14409.png', '1635.png', '10121.png', '7278.png', '1806.png', '2489.png', '4991.png',
                      '7240.png', '3740.png', '2304.png', '3768.png', '5133.png', '1433.png', '11777.png', '6348.png',
                      '2660.png', '1584.png', '1590.png', '14960.png', '11830.png', '14009.png', '15317.png',
                      '6200.png', '1786.png', '3634.png', '1989.png', '14551.png', '14544.png', '1778.png', '2259.png',
                      '1744.png', '7484.png', '3184.png', '1787.png', '2067.png', '9668.png', '7889.png', '4358.png',
                      '2098.png', '3386.png', '5085.png', '1585.png', '6413.png', '6407.png', '7731.png', '3543.png',
                      '8370.png', '11212.png', '1630.png', '1817.png', '3794.png', '1829.png', '1815.png', '11210.png',
                      '3743.png', '3025.png', '4776.png', '1381.png', '3555.png', '1430.png', '3541.png', '8115.png',
                      '12797.png', '10180.png', '12241.png', '10456.png', '6363.png', '4212.png', '3384.png',
                      '5087.png', '10278.png', '7121.png', '1550.png', '1544.png', '3435.png', '1578.png', '2298.png',
                      '5285.png', '3623.png', '4616.png', '8992.png', '2273.png', '6014.png', '14547.png', '15117.png',
                      '6028.png', '8011.png', '1753.png', '8777.png', '1974.png', '3839.png', '1545.png', '11167.png',
                      '7120.png', '3346.png', '1586.png', '1592.png', '2104.png', '4213.png', '2662.png', '3568.png',
                      '2110.png', '9092.png', '12283.png', '15416.png', '2689.png', '3018.png', '3742.png', '14369.png',
                      '4978.png', '8824.png', '4089.png', '1898.png', '14865.png', '8314.png', '13677.png', '4248.png',
                      '14939.png', '2017.png', '1913.png', '12192.png', '1907.png', '2201.png', '4116.png', '3645.png',
                      '3889.png', '1720.png', '3644.png', '4103.png', '4665.png', '13528.png', '3678.png', '2994.png',
                      '2002.png', '2943.png', '7185.png', '14086.png', '9194.png', '3526.png', '3532.png', '13312.png',
                      '2438.png', '9579.png', '1643.png', '12196.png', '1496.png', '3256.png', '11711.png', '3493.png',
                      '1521.png', '1723.png', '4883.png', '5223.png', '1737.png', '5236.png', '8706.png', '3445.png',
                      '1508.png', '3479.png', '2773.png', '1454.png', '1497.png', '9785.png', '2439.png', '4935.png',
                      '1849.png', '3079.png', '4064.png', '1493.png', '2159.png', '1444.png', '3253.png', '6303.png',
                      '4272.png', '2617.png', '11106.png', '14724.png', '1929.png', '12816.png', '1915.png', '5554.png',
                      '2575.png', '9435.png', '10755.png', '3642.png', '3865.png', '3681.png', '14043.png', '12354.png',
                      '1519.png', '3468.png', '2038.png', '2992.png', '15375.png', '7815.png', '3497.png', '1445.png',
                      '3520.png', '9970.png', '4071.png', '1647.png', '3050.png', '1684.png', '7593.png', '3939.png',
                      '1692.png', '12791.png', '1645.png', '2358.png', '12593.png', '6328.png', '1447.png', '11717.png',
                      '8463.png', '3481.png', '13074.png', '3456.png', '2006.png', '2774.png', '10780.png', '1719.png',
                      '14242.png', '8728.png', '4112.png', '1730.png', '8927.png', '1917.png', '11892.png', '14726.png',
                      '15216.png', '5185.png', '4072.png', '1877.png', '1863.png', '14493.png', '4080.png', '12004.png',
                      '1885.png', '14475.png', '5349.png', '1891.png', '1488.png', '7990.png', '13118.png', '3274.png',
                      '1477.png', '11727.png', '4269.png', '2787.png', '14071.png', '10798.png', '4137.png', '1714.png',
                      '6053.png', '5228.png', '1927.png', '2584.png', '3301.png', '5770.png', '2962.png', '2786.png',
                      '6286.png', '4240.png', '2157.png', '12207.png', '2180.png', '2194.png', '5374.png', '15024.png',
                      '9228.png', '8678.png', '1879.png', '8938.png', '1886.png', '3061.png', '13443.png', '1448.png',
                      '9000.png', '15542.png', '2948.png', '7830.png', '4322.png', '1500.png', '3303.png', '1919.png',
                      '12832.png', '4120.png', '1702.png', '4108.png', '1716.png', '3114.png', '4135.png', '8082.png',
                      '3855.png', '1515.png', '1501.png', '1529.png', '2838.png', '11043.png', '3289.png', '8647.png',
                      '1663.png', '1677.png', '4041.png', '9571.png', '13495.png', '1850.png', '13491.png', '1673.png',
                      '9788.png', '1471.png', '8441.png', '6322.png', '2622.png', '1459.png', '7809.png', '2965.png',
                      '13095.png', '11682.png', '1505.png', '3460.png', '2030.png', '7956.png', '1934.png', '15157.png',
                      '1707.png', '14261.png', '10760.png', '3887.png', '9366.png', '14512.png', '3663.png', '6900.png',
                      '4865.png', '2743.png', '2757.png', '3515.png', '13645.png', '9038.png', '11085.png', '1896.png',
                      '12759.png', '3703.png', '8587.png', '3926.png', '2379.png', '12997.png', '1658.png', '1499.png',
                      '7942.png', '3517.png', '1472.png', '14672.png', '6321.png', '1512.png', '13055.png', '3339.png',
                      '12149.png', '1704.png', '2231.png', '4133.png', '1711.png', '14539.png', '3847.png', '1936.png',
                      '10788.png', '2032.png', '1507.png', '8245.png', '7758.png', '9013.png', '1473.png', '2813.png',
                      '1895.png', '4047.png', '13487.png', '12941.png', '7515.png', '1399.png', '4222.png', '10300.png',
                      '7717.png', '2082.png', '2727.png', '14950.png', '12305.png', '3377.png', '12339.png', '1945.png',
                      '1979.png', '1992.png', '2256.png', '11369.png', '2242.png', '12853.png', '1944.png', '13768.png',
                      '2083.png', '4579.png', '14167.png', '9707.png', '1398.png', '7266.png', '9539.png', '2322.png',
                      '15079.png', '1629.png', '3994.png', '7927.png', '12299.png', '4235.png', '1588.png', '3406.png',
                      '5077.png', '1952.png', '1985.png', '1761.png', '1775.png', '3163.png', '6769.png', '10048.png',
                      '1576.png', '15454.png', '2725.png', '2057.png', '2731.png', '4340.png', '13780.png', '5089.png',
                      '13145.png', '1402.png', '4208.png', '2137.png', '3229.png', '13186.png', '4585.png', '2309.png',
                      '10672.png', '1833.png', '12920.png', '7507.png', '4740.png', '8387.png', '2682.png', '3588.png',
                      '1406.png', '2669.png', '14160.png', '2084.png', '9660.png', '2053.png', '3359.png', '1943.png',
                      '4185.png', '13551.png', '5270.png', '1980.png', '12672.png', '10071.png', '7300.png',
                      '11421.png', '3166.png', '3614.png', '6234.png', '1407.png', '2668.png', '3403.png', '15041.png',
                      '7510.png', '2454.png', '6626.png', '2440.png', '8582.png', '1565.png', '2736.png', '3825.png',
                      '1940.png', '1767.png', '1997.png', '12664.png', '11422.png', '1941.png', '1955.png', '2045.png',
                      '3415.png', '5919.png', '2858.png', '3987.png', '12077.png', '1612.png', '2482.png', '4971.png',
                      '3381.png', '7130.png', '13204.png', '10042.png', '8564.png', '14152.png', '12443.png',
                      '2934.png', '8821.png', '13830.png', '13824.png', '10336.png', '5718.png', '10242.png',
                      '14024.png', '8994.png', '10727.png', '5282.png', '11607.png', '5137.png', '8836.png', '3026.png',
                      '8371.png', '9696.png', '15114.png', '10087.png', '12684.png', '6573.png', '4210.png',
                      '11762.png', '5867.png', '4984.png', '14234.png', '7322.png', '1683.png', '12838.png', '9619.png',
                      '10237.png', '11075.png', '7542.png', '5035.png', '8061.png', '1939.png', '8921.png', '10552.png',
                      '13649.png', '7555.png', '8128.png', '8851.png', '7784.png', '9024.png', '5959.png', '14042.png',
                      '10224.png', '6062.png', '7425.png', '5569.png', '1733.png', '14269.png', '7626.png', '15413.png',
                      '2372.png', '3911.png', '13458.png', '14860.png', '11663.png', '8067.png', '3325.png', '5016.png',
                      '10385.png', '11500.png', '13326.png', '7562.png', '13130.png', '14918.png', '12819.png',
                      '14265.png', '2544.png', '6709.png', '7826.png', '10362.png', '9943.png', '8108.png', '6643.png',
                      '8451.png', '11917.png', '11863.png', '10758.png', '8055.png', '5997.png', '4243.png', '5834.png',
                      '13481.png', '3935.png', '7570.png', '6450.png', '5211.png', '9401.png', '13719.png', '9762.png',
                      '15234.png', '12003.png', '4722.png', '7559.png', '8124.png', '5366.png', '9548.png', '10658.png',
                      '6490.png', '6484.png', '1466.png', '9833.png', '7822.png', '15430.png', '4126.png', '3884.png',
                      '4325.png', '11119.png', '5507.png', '5934.png', '13726.png', '14049.png', '5946.png',
                      '11910.png', '8872.png', '8140.png', '3388.png', '14012.png', '7884.png', '14011.png', '2687.png',
                      '8785.png', '7449.png', '11619.png', '9908.png', '8633.png', '11235.png', '7112.png', '2296.png',
                      '5937.png', '11222.png', '3003.png', '9258.png', '9516.png', '7249.png', '9728.png', '13974.png',
                      '8218.png', '9489.png', '8967.png', '15479.png', '8225.png', '12074.png', '5306.png', '3213.png',
                      '15069.png', '9271.png', '15321.png', '4796.png', '6154.png', '4031.png', '14836.png',
                      '11769.png', '7048.png', '9893.png', '8233.png', '10067.png', '10715.png', '1766.png', '6784.png',
                      '1969.png', '11387.png', '9110.png', '7840.png', '2938.png', '14604.png', '9919.png']

# bad_segmented
# ['4203.png', '3544.png', '2909.png', '1596.png', '1582.png', '2060.png', '2706.png', '11605.png', '3430.png', '10043.png', '12898.png', '5518.png', '4174.png', '12866.png', '1396.png', '13013.png', '5054.png', '2061.png', '3551.png', '8438.png', '10491.png', '1636.png', '3021.png', '2471.png', '6824.png', '3976.png', '1387.png', '8599.png', '5917.png', '7696.png', '8205.png', '3369.png', '2705.png', '3433.png', '15312.png', '3816.png', '1973.png', '8980.png', '4163.png', '3181.png', '8573.png', '3354.png', '11613.png', '3432.png', '2062.png', '9862.png', '4229.png', '9057.png', '11983.png', '3585.png', '10653.png', '14353.png', '4017.png', '4942.png', '3797.png', '1625.png', '4007.png', '5682.png', '3595.png', '5127.png', '3556.png', '10441.png', '2066.png', '3378.png', '7491.png', '3813.png', '3621.png', '3635.png', '7453.png', '3812.png', '4199.png', '4204.png', '2107.png', '4576.png', '1426.png', '3225.png', '1432.png', '1624.png', '10118.png', '1803.png', '3958.png', '3970.png', '7928.png', '14817.png', '2139.png', '3569.png', '4206.png', '3848.png', '7796.png', '7184.png', '1593.png', '2059.png', '13017.png', '14022.png', '6203.png', '2065.png', '2071.png', '12447.png', '3637.png', '10720.png', '14546.png', '4164.png', '14552.png', '14235.png', '3811.png', '3187.png', '6558.png', '1431.png', '11328.png', '3583.png', '9535.png', '2312.png', '3756.png', '1814.png', '12917.png', '9591.png', '1873.png', '8102.png', '1640.png', '3731.png', '1654.png', '1481.png', '14093.png', '5793.png', '2765.png', '1522.png', '3692.png', '3679.png', '3650.png', '2572.png', '10035.png', '5584.png', '10948.png', '3863.png', '13702.png', '11667.png', '3491.png', '3485.png', '13306.png', '1899.png', '3730.png', '3917.png', '1870.png', '13489.png', '8840.png', '1864.png', '10816.png', '9545.png', '3040.png', '13476.png', '13304.png', '4288.png', '3518.png', '1469.png', '3524.png', '1455.png', '9182.png', '3487.png', '4303.png', '2014.png', '2000.png', '3478.png', '5747.png', '3444.png', '2982.png', '3875.png', '1910.png', '3685.png', '12813.png', '2216.png', '2558.png', '7421.png', '1736.png', '1722.png', '12153.png', '2571.png', '14287.png', '1911.png', '1534.png', '2983.png', '2029.png', '12351.png', '6515.png', '3486.png', '3492.png', '2175.png', '3294.png', '4289.png', '1656.png', '2363.png', '2405.png', '3928.png', '8699.png', '13315.png', '3051.png', '5346.png', '12023.png', '11099.png', '5150.png', '2165.png', '9742.png', '3509.png', '1478.png', '7155.png', '1518.png', '9391.png', '3657.png', '4138.png', '2213.png', '9352.png', '5568.png', '10026.png', '11338.png', '3124.png', '3871.png', '1928.png', '3695.png', '6504.png', '10219.png', '1531.png', '15407.png', '10594.png', '11852.png', '1451.png', '9794.png', '4298.png', '6666.png', '2428.png', '8844.png', '1874.png', '10186.png', '15015.png', '11273.png', '1679.png', '2364.png', '2827.png', '6466.png', '3495.png', '3324.png', '7624.png', '12418.png', '1527.png', '4305.png', '2012.png', '7426.png', '3898.png', '7340.png', '14525.png', '2204.png', '2210.png', '7369.png', '1718.png', '11474.png', '1724.png', '3655.png', '6088.png', '2588.png', '3682.png', '2013.png', '10540.png', '3457.png', '2985.png', '3494.png', '7779.png', '9026.png', '2173.png', '2365.png', '8890.png', '10636.png', '1650.png', '3053.png', '4043.png', '15019.png', '2340.png', '3076.png', '7576.png', '1675.png', '11041.png', '2022.png', '2585.png', '1729.png', '7358.png', '6720.png', '3880.png', '1701.png', '1700.png', '1502.png', '3473.png', '1516.png', '10558.png', '7007.png', '1476.png', '2143.png', '3249.png', '14689.png', '1489.png', '12987.png', '3711.png', '2341.png', '1648.png', '1884.png', '3921.png', '4068.png', '1662.png', '4295.png', '8493.png', '4256.png', '10214.png', '15378.png', '12417.png', '8901.png', '4875.png', '3897.png', '3100.png', '1717.png', '3672.png', '2222.png', '12600.png', '4121.png', '3896.png', '6905.png', '1918.png', '11137.png', '13708.png', '6534.png', '2034.png', '11057.png', '11527.png', '14463.png', '1844.png', '3919.png', '8680.png', '8858.png', '3925.png', '6691.png', '11537.png', '2352.png', '12002.png', '2178.png', '3514.png', '3528.png', '9159.png', '1511.png', '14711.png', '4327.png', '3448.png', '4333.png', '3674.png', '7389.png', '8910.png', '3110.png', '1713.png', '3104.png', '2232.png', '2226.png', '3886.png', '6041.png', '2541.png', '3688.png', '15181.png', '3449.png', '4326.png', '3475.png', '7029.png', '1458.png', '2623.png', '4285.png', '2192.png', '10628.png', '14328.png', '1672.png', '13484.png', '4085.png', '3715.png', '1664.png', '1880.png', '13690.png', '4293.png', '12559.png', '2153.png', '12217.png', '3477.png', '2027.png', '2755.png', '10574.png', '1923.png', '6094.png', '8907.png', '5210.png', '3675.png', '3649.png', '1922.png', '6917.png', '2754.png', '3338.png', '3476.png', '3462.png', '12570.png', '7994.png', '4279.png', '14868.png', '4286.png', '3728.png', '1659.png', '4090.png', '6877.png', '9289.png', '2337.png', '3983.png', '3767.png', '13184.png', '6353.png', '11988.png', '2096.png', '2733.png', '3363.png', '3820.png', '4154.png', '4168.png', '5505.png', '3174.png', '1763.png', '3606.png', '2518.png', '1987.png', '4627.png', '4141.png', '11382.png', '3821.png', '9883.png', '11023.png', '3564.png', '8431.png', '4237.png', '8419.png', '2861.png', '3766.png', '3969.png', '1830.png', '11579.png', '3764.png', '1601.png', '15051.png', '2650.png', '2136.png', '2056.png', '1563.png', '3837.png', '13540.png', '5513.png', '3611.png', '1748.png', '5248.png', '10074.png', '14212.png', '8977.png', '6796.png', '3836.png', '4354.png', '14010.png', '15497.png', '3573.png', '8368.png', '1600.png', '8142.png', '3771.png', '1614.png', '3759.png', '3956.png', '3007.png', '3013.png', '6382.png', '9072.png', '2133.png', '2655.png', '13784.png', '6209.png', '10500.png', '1764.png', '8740.png', '5516.png', '1995.png', '3628.png', '5259.png', '1956.png', '3833.png', '13593.png', '5729.png', '6546.png', '5073.png', '2085.png', '5305.png', '13424.png', '1605.png', '3012.png', '10852.png', '8192.png', '3951.png', '1607.png', '13426.png', '3004.png', '6632.png', '1388.png', '7921.png', '3574.png', '3212.png', '4569.png', '13156.png', '2130.png', '5893.png', '2642.png', '2939.png', '2087.png', '2078.png', '1559.png', '1798.png', '3616.png', '4179.png', '4623.png', '10927.png', '12301.png', '13989.png', '2086.png', '7049.png', '11740.png', '3207.png', '1404.png', '2327.png', '5448.png', '9266.png', '8150.png', '4018.png', '1835.png', '3978.png']
# to_rotate_or_flip
# ['7871.png', '1804.png', '3961.png', '3949.png', '3791.png', '1838.png', '3785.png', '9525.png', '2316.png', '6164.png', '10645.png', '4773.png', '6602.png', '3752.png', '1637.png', '5860.png', '7911.png', '10484.png', '7093.png', '2882.png', '11765.png', '12495.png', '13947.png', '7118.png', '9109.png', '1555.png', '10902.png', '1958.png', '12867.png', '6762.png', '4613.png', '2262.png', '9323.png', '3168.png', '4161.png', '7468.png', '8983.png', '8773.png', '1743.png', '8001.png', '7454.png', '11375.png', '3632.png', '5242.png', '8772.png', '13211.png', '1756.png', '8982.png', '4160.png', '2288.png', '1781.png', '1959.png', '6788.png', '3828.png', '8969.png', '1965.png', '1971.png', '11176.png', '3431.png', '5726.png', '2049.png', '1554.png', '11604.png', '11162.png', '6213.png', '12325.png', '1568.png', '6561.png', '2707.png', '11837.png', '2883.png', '1434.png', '2129.png', '2667.png', '15258.png', '12537.png', '14620.png', '6415.png', '12523.png', '1385.png', '3592.png', '2868.png', '8606.png', '8148.png', '4000.png', '7290.png', '10875.png', '3779.png', '12723.png', '3989.png', '7537.png', '15070.png', '11969.png', '1378.png', '1422.png', '14781.png', '1581.png', '4166.png', '2077.png', '12327.png', '12441.png', '3341.png', '1556.png', '10083.png', '1783.png', '4189.png', '2513.png', '4162.png', '14232.png', '6013.png', '1768.png', '9334.png', '4176.png', '5240.png', '1998.png', '15104.png', '7324.png', '3630.png', '12642.png', '10055.png', '9321.png', '1782.png', '1972.png', '1966.png', '5057.png', '2076.png', '9876.png', '2923.png', '1594.png', '2937.png', '1437.png', '3591.png', '14409.png', '1635.png', '10121.png', '7278.png', '1806.png', '2489.png', '4991.png', '7240.png', '3740.png', '2304.png', '3768.png', '5133.png', '1433.png', '11777.png', '6348.png', '2660.png', '1584.png', '1590.png', '14960.png', '11830.png', '14009.png', '15317.png', '6200.png', '1786.png', '3634.png', '1989.png', '14551.png', '14544.png', '1778.png', '2259.png', '1744.png', '7484.png', '3184.png', '1787.png', '2067.png', '9668.png', '7889.png', '4358.png', '2098.png', '3386.png', '5085.png', '1585.png', '6413.png', '6407.png', '7731.png', '3543.png', '8370.png', '11212.png', '1630.png', '1817.png', '3794.png', '1829.png', '1815.png', '11210.png', '3743.png', '3025.png', '4776.png', '1381.png', '3555.png', '1430.png', '3541.png', '8115.png', '12797.png', '10180.png', '12241.png', '10456.png', '6363.png', '4212.png', '3384.png', '5087.png', '10278.png', '7121.png', '1550.png', '1544.png', '3435.png', '1578.png', '2298.png', '5285.png', '3623.png', '4616.png', '8992.png', '2273.png', '6014.png', '14547.png', '15117.png', '6028.png', '8011.png', '1753.png', '8777.png', '1974.png', '3839.png', '1545.png', '11167.png', '7120.png', '3346.png', '1586.png', '1592.png', '2104.png', '4213.png', '2662.png', '3568.png', '2110.png', '9092.png', '12283.png', '15416.png', '2689.png', '3018.png', '3742.png', '14369.png', '4978.png', '8824.png', '4089.png', '1898.png', '14865.png', '8314.png', '13677.png', '4248.png', '14939.png', '2017.png', '1913.png', '12192.png', '1907.png', '2201.png', '4116.png', '3645.png', '3889.png', '1720.png', '3644.png', '4103.png', '4665.png', '13528.png', '3678.png', '2994.png', '2002.png', '2943.png', '7185.png', '14086.png', '9194.png', '3526.png', '3532.png', '13312.png', '2438.png', '9579.png', '1643.png', '12196.png', '1496.png', '3256.png', '11711.png', '3493.png', '1521.png', '1723.png', '4883.png', '5223.png', '1737.png', '5236.png', '8706.png', '3445.png', '1508.png', '3479.png', '2773.png', '1454.png', '1497.png', '9785.png', '2439.png', '4935.png', '1849.png', '3079.png', '4064.png', '1493.png', '2159.png', '1444.png', '3253.png', '6303.png', '4272.png', '2617.png', '11106.png', '14724.png', '1929.png', '12816.png', '1915.png', '5554.png', '2575.png', '9435.png', '10755.png', '3642.png', '3865.png', '3681.png', '14043.png', '12354.png', '1519.png', '3468.png', '2038.png', '2992.png', '15375.png', '7815.png', '3497.png', '1445.png', '3520.png', '9970.png', '4071.png', '1647.png', '3050.png', '1684.png', '7593.png', '3939.png', '1692.png', '12791.png', '1645.png', '2358.png', '12593.png', '6328.png', '1447.png', '11717.png', '8463.png', '3481.png', '13074.png', '3456.png', '2006.png', '2774.png', '10780.png', '1719.png', '14242.png', '8728.png', '4112.png', '1730.png', '8927.png', '1917.png', '11892.png', '14726.png', '15216.png', '5185.png', '4072.png', '1877.png', '1863.png', '14493.png', '4080.png', '12004.png', '1885.png', '14475.png', '5349.png', '1891.png', '1488.png', '7990.png', '13118.png', '3274.png', '1477.png', '11727.png', '4269.png', '2787.png', '14071.png', '10798.png', '4137.png', '1714.png', '6053.png', '5228.png', '1927.png', '2584.png', '3301.png', '5770.png', '2962.png', '2786.png', '6286.png', '4240.png', '2157.png', '12207.png', '2180.png', '2194.png', '5374.png', '15024.png', '9228.png', '8678.png', '1879.png', '8938.png', '1886.png', '3061.png', '13443.png', '1448.png', '9000.png', '15542.png', '2948.png', '7830.png', '4322.png', '1500.png', '3303.png', '1919.png', '12832.png', '4120.png', '1702.png', '4108.png', '1716.png', '3114.png', '4135.png', '8082.png', '3855.png', '1515.png', '1501.png', '1529.png', '2838.png', '11043.png', '3289.png', '8647.png', '1663.png', '1677.png', '4041.png', '9571.png', '13495.png', '1850.png', '13491.png', '1673.png', '9788.png', '1471.png', '8441.png', '6322.png', '2622.png', '1459.png', '7809.png', '2965.png', '13095.png', '11682.png', '1505.png', '3460.png', '2030.png', '7956.png', '1934.png', '15157.png', '1707.png', '14261.png', '10760.png', '3887.png', '9366.png', '14512.png', '3663.png', '6900.png', '4865.png', '2743.png', '2757.png', '3515.png', '13645.png', '9038.png', '11085.png', '1896.png', '12759.png', '3703.png', '8587.png', '3926.png', '2379.png', '12997.png', '1658.png', '1499.png', '7942.png', '3517.png', '1472.png', '14672.png', '6321.png', '1512.png', '13055.png', '3339.png', '12149.png', '1704.png', '2231.png', '4133.png', '1711.png', '14539.png', '3847.png', '1936.png', '10788.png', '2032.png', '1507.png', '8245.png', '7758.png', '9013.png', '1473.png', '2813.png', '1895.png', '4047.png', '13487.png', '12941.png', '7515.png', '1399.png', '4222.png', '10300.png', '7717.png', '2082.png', '2727.png', '14950.png', '12305.png', '3377.png', '12339.png', '1945.png', '1979.png', '1992.png', '2256.png', '11369.png', '2242.png', '12853.png', '1944.png', '13768.png', '2083.png', '4579.png', '14167.png', '9707.png', '1398.png', '7266.png', '9539.png', '2322.png', '15079.png', '1629.png', '3994.png', '7927.png', '12299.png', '4235.png', '1588.png', '3406.png', '5077.png', '1952.png', '1985.png', '1761.png', '1775.png', '3163.png', '6769.png', '10048.png', '1576.png', '15454.png', '2725.png', '2057.png', '2731.png', '4340.png', '13780.png', '5089.png', '13145.png', '1402.png', '4208.png', '2137.png', '3229.png', '13186.png', '4585.png', '2309.png', '10672.png', '1833.png', '12920.png', '7507.png', '4740.png', '8387.png', '2682.png', '3588.png', '1406.png', '2669.png', '14160.png', '2084.png', '9660.png', '2053.png', '3359.png', '1943.png', '4185.png', '13551.png', '5270.png', '1980.png', '12672.png', '10071.png', '7300.png', '11421.png', '3166.png', '3614.png', '6234.png', '1407.png', '2668.png', '3403.png', '15041.png', '7510.png', '2454.png', '6626.png', '2440.png', '8582.png', '1565.png', '2736.png', '3825.png', '1940.png', '1767.png', '1997.png', '12664.png', '11422.png', '1941.png', '1955.png', '2045.png', '3415.png', '5919.png', '2858.png', '3987.png', '12077.png', '1612.png', '2482.png', '4971.png']
# particolare
# '3381.png', '7130.png', '13204.png', '10042.png', '8564.png', '14152.png', '12443.png', '2934.png', '8821.png', '13830.png', '13824.png', '10336.png', '5718.png', '10242.png', '14024.png', '8994.png', '10727.png', '5282.png', '11607.png', '5137.png', '8836.png', '3026.png', '8371.png', '9696.png', '15114.png', '10087.png', '12684.png', '6573.png', '4210.png', '11762.png', '5867.png', '4984.png', '14234.png', '7322.png', '1683.png', '12838.png', '9619.png', '10237.png', '11075.png', '7542.png', '5035.png', '8061.png', '1939.png', '8921.png', '10552.png', '13649.png', '7555.png', '8128.png', '8851.png', '7784.png', '9024.png', '5959.png', '14042.png', '10224.png', '6062.png', '7425.png', '5569.png', '1733.png', '14269.png', '7626.png', '15413.png', '2372.png', '3911.png', '13458.png', '14860.png', '11663.png', '8067.png', '3325.png', '5016.png', '10385.png', '11500.png', '13326.png', '7562.png', '13130.png', '14918.png', '12819.png', '14265.png', '2544.png', '6709.png', '7826.png', '10362.png', '9943.png', '8108.png', '6643.png', '8451.png', '11917.png', '11863.png', '10758.png', '8055.png', '5997.png', '4243.png', '5834.png', '13481.png', '3935.png', '7570.png', '6450.png', '5211.png', '9401.png', '13719.png', '9762.png', '15234.png', '12003.png', '4722.png', '7559.png', '8124.png', '5366.png', '9548.png', '10658.png', '6490.png', '6484.png', '1466.png', '9833.png', '7822.png', '15430.png', '4126.png', '3884.png', '4325.png', '11119.png', '5507.png', '5934.png', '13726.png', '14049.png', '5946.png', '11910.png', '8872.png', '8140.png', '3388.png', '14012.png', '7884.png', '14011.png', '2687.png', '8785.png', '7449.png', '11619.png', '9908.png', '8633.png', '11235.png', '7112.png', '2296.png', '5937.png', '11222.png', '3003.png', '9258.png', '9516.png', '7249.png', '9728.png', '13974.png', '8218.png', '9489.png', '8967.png', '15479.png', '8225.png', '12074.png', '5306.png', '3213.png', '15069.png', '9271.png', '15321.png', '4796.png', '6154.png', '4031.png', '14836.png', '11769.png', '7048.png', '9893.png', '8233.png', '10067.png', '10715.png', '1766.png', '6784.png', '1969.png', '11387.png', '9110.png', '7840.png', '2938.png', '14604.png', '9919.png']

# 500
# bad_segmented = ['4203.png', '3544.png', '2909.png', '1596.png', '1582.png', '2060.png', '2706.png', '11605.png', '3430.png', '10043.png', '12898.png', '5518.png', '4174.png', '12866.png']
# to_rotate_or_flip = ['7871.png', '1804.png', '3961.png', '3949.png', '3791.png', '1838.png', '3785.png', '9525.png', '2316.png', '6164.png', '10645.png', '4773.png', '6602.png', '3752.png', '1637.png', '5860.png', '7911.png', '10484.png', '7093.png', '2882.png', '11765.png', '12495.png', '13947.png', '7118.png', '9109.png', '1555.png', '10902.png', '1958.png', '12867.png', '6762.png', '4613.png', '2262.png', '9323.png', '3168.png', '4161.png', '7468.png', '8983.png', '8773.png', '1743.png', '8001.png', '7454.png', '11375.png', '3632.png', '5242.png', '8772.png', '13211.png', '1756.png', '8982.png', '4160.png', '2288.png', '1781.png', '1959.png', '6788.png', '3828.png', '8969.png', '1965.png', '1971.png']
# particolare = ['3381.png', '7130.png', '13204.png', '10042.png', '8564.png']

# 1000
# bad_segmented = ['1396.png', '13013.png', '5054.png', '2061.png', '3551.png', '8438.png', '10491.png', '1636.png', '3021.png', '2471.png', '6824.png', '3976.png', '1387.png', '8599.png', '5917.png', '7696.png']
# to_rotate_or_flip = ['11176.png', '3431.png', '5726.png', '2049.png', '1554.png', '11604.png', '11162.png', '6213.png', '12325.png', '1568.png', '6561.png', '2707.png', '11837.png', '2883.png', '1434.png', '2129.png', '2667.png', '15258.png', '12537.png', '14620.png', '6415.png', '12523.png', '1385.png', '3592.png', '2868.png', '8606.png', '8148.png', '4000.png', '7290.png', '10875.png', '3779.png', '12723.png', '3989.png', '7537.png', '15070.png', '11969.png', '1378.png', '1422.png', '14781.png', '1581.png']
# particolare = ['14152.png', '12443.png', '2934.png', '8821.png', '13830.png', '13824.png', '10336.png']

# 1500
# bad_segmented = ['8205.png', '3369.png', '2705.png', '3433.png', '15312.png', '3816.png', '1973.png', '8980.png', '4163.png', '3181.png', '8573.png', '3354.png', '11613.png', '3432.png', '2062.png', '9862.png', '4229.png', '9057.png', '11983.png', '3585.png', '10653.png']
# to_rotate_or_flip = ['4166.png', '2077.png', '12327.png', '12441.png', '3341.png', '1556.png', '10083.png', '1783.png', '4189.png', '2513.png', '4162.png', '14232.png', '6013.png', '1768.png', '9334.png', '4176.png', '5240.png', '1998.png', '15104.png', '7324.png', '3630.png', '12642.png', '10055.png', '9321.png', '1782.png', '1972.png', '1966.png', '5057.png', '2076.png', '9876.png', '2923.png', '1594.png', '2937.png', '1437.png', '3591.png', '14409.png', '1635.png']
# particolare = ['5718.png', '10242.png', '14024.png', '8994.png', '10727.png', '5282.png', '11607.png', '5137.png']

# 2000
# bad_segmented = ['14353.png', '4017.png', '4942.png', '3797.png', '1625.png', '4007.png', '5682.png', '3595.png', '5127.png', '3556.png', '10441.png', '2066.png', '3378.png', '7491.png', '3813.png', '3621.png']
# to_rotate_or_flip = ['10121.png', '7278.png', '1806.png', '2489.png', '4991.png', '7240.png', '3740.png', '2304.png', '3768.png', '5133.png', '1433.png', '11777.png', '6348.png', '2660.png', '1584.png', '1590.png', '14960.png', '11830.png', '14009.png', '15317.png', '6200.png', '1786.png', '3634.png', '1989.png', '14551.png', '14544.png', '1778.png', '2259.png', '1744.png']
# particolare = ['8836.png', '3026.png', '8371.png', '9696.png']

# 2500
# bad_segmented = ['3635.png', '7453.png', '3812.png', '4199.png', '4204.png', '2107.png', '4576.png', '1426.png', '3225.png', '1432.png', '1624.png', '10118.png', '1803.png', '3958.png', '3970.png', '7928.png', '14817.png', '2139.png', '3569.png', '4206.png']
# to_rotate_or_flip = ['7484.png', '3184.png', '1787.png', '2067.png', '9668.png', '7889.png', '4358.png', '2098.png', '3386.png', '5085.png', '1585.png', '6413.png', '6407.png', '7731.png', '3543.png', '8370.png', '11212.png', '1630.png', '1817.png', '3794.png', '1829.png', '1815.png', '11210.png', '3743.png', '3025.png', '4776.png', '1381.png', '3555.png', '1430.png', '3541.png']
# particolare = ['15114.png', '10087.png', '12684.png', '6573.png', '4210.png', '11762.png', '5867.png', '4984.png']

# 3000
# bad_segmented = ['3848.png', '7796.png', '7184.png', '1593.png', '2059.png', '13017.png', '14022.png', '6203.png', '2065.png', '2071.png', '12447.png', '3637.png', '10720.png', '14546.png', '4164.png', '14552.png', '14235.png', '3811.png', '3187.png', '6558.png', '1431.png']
# to_rotate_or_flip = ['8115.png', '12797.png', '10180.png', '12241.png', '10456.png', '6363.png', '4212.png', '3384.png', '5087.png', '10278.png', '7121.png', '1550.png', '1544.png', '3435.png', '1578.png', '2298.png', '5285.png', '3623.png', '4616.png', '8992.png', '2273.png', '6014.png', '14547.png', '15117.png', '6028.png', '8011.png', '1753.png', '8777.png', '1974.png', '3839.png', '1545.png', '11167.png', '7120.png', '3346.png', '1586.png', '1592.png', '2104.png', '4213.png', '2662.png', '3568.png', '2110.png', '9092.png', '12283.png']
# particolare = ['14234.png', '7322.png']

# 4000
# bad_segmented = ['11328.png', '3583.png', '9535.png', '2312.png', '3756.png', '1814.png', '12917.png', '9591.png', '1873.png', '8102.png', '1640.png', '3731.png', '1654.png', '1481.png', '14093.png', '5793.png', '2765.png', '1522.png', '3692.png', '3679.png', '3650.png', '2572.png', '10035.png', '5584.png', '10948.png', '3863.png', '13702.png', '11667.png', '3491.png', '3485.png', '13306.png', '1899.png', '3730.png', '3917.png', '1870.png', '13489.png', '8840.png', '1864.png', '10816.png', '9545.png', '3040.png', '13476.png', '13304.png']
# to_rotate_or_flip = ['15416.png', '2689.png', '3018.png', '3742.png', '14369.png', '4978.png', '8824.png', '4089.png', '1898.png', '14865.png', '8314.png', '13677.png', '4248.png', '14939.png', '2017.png', '1913.png', '12192.png', '1907.png', '2201.png', '4116.png', '3645.png', '3889.png', '1720.png', '3644.png', '4103.png', '4665.png', '13528.png', '3678.png', '2994.png', '2002.png', '2943.png', '7185.png', '14086.png', '9194.png', '3526.png', '3532.png', '13312.png', '2438.png', '9579.png', '1643.png']
# particolare = ['1683.png', '12838.png', '9619.png', '10237.png', '11075.png', '7542.png']

# 5000
# bad_segmented = ['4288.png', '3518.png', '1469.png', '3524.png', '1455.png', '9182.png', '3487.png', '4303.png', '2014.png', '2000.png', '3478.png', '5747.png', '3444.png', '2982.png', '3875.png', '1910.png', '3685.png', '12813.png', '2216.png', '2558.png', '7421.png', '1736.png', '1722.png', '12153.png', '2571.png', '14287.png', '1911.png', '1534.png', '2983.png', '2029.png', '12351.png', '6515.png', '3486.png', '3492.png', '2175.png', '3294.png', '4289.png', '1656.png', '2363.png', '2405.png', '3928.png', '8699.png', '13315.png', '3051.png', '5346.png', '12023.png', '11099.png', '5150.png', '2165.png', '9742.png', '3509.png', '1478.png', '7155.png', '1518.png', '9391.png', '3657.png', '4138.png']
# to_rotate_or_flip = ['12196.png', '1496.png', '3256.png', '11711.png', '3493.png', '1521.png', '1723.png', '4883.png', '5223.png', '1737.png', '5236.png', '8706.png', '3445.png', '1508.png', '3479.png', '2773.png', '1454.png', '1497.png', '9785.png', '2439.png', '4935.png', '1849.png', '3079.png', '4064.png', '1493.png', '2159.png', '1444.png', '3253.png', '6303.png', '4272.png', '2617.png', '11106.png', '14724.png', '1929.png', '12816.png', '1915.png']
# particolare = ['5035.png', '8061.png', '1939.png', '8921.png', '10552.png', '13649.png', '7555.png', '8128.png', '8851.png', '7784.png', '9024.png', '5959.png', '14042.png', '10224.png']

# 5500
# bad_segmented = ['2213.png', '9352.png', '5568.png', '10026.png', '11338.png', '3124.png', '3871.png', '1928.png', '3695.png', '6504.png', '10219.png', '1531.png', '15407.png', '10594.png', '11852.png', '1451.png', '9794.png', '4298.png', '6666.png', '2428.png', '8844.png', '1874.png', '10186.png', '15015.png', '11273.png']
# to_rotate_or_flip = ['5554.png', '2575.png', '9435.png', '10755.png', '3642.png', '3865.png', '3681.png', '14043.png', '12354.png', '1519.png', '3468.png', '2038.png', '2992.png', '15375.png', '7815.png', '3497.png', '1445.png', '3520.png', '9970.png', '4071.png', '1647.png', '3050.png', '1684.png', '7593.png', '3939.png', '1692.png', '12791.png', '1645.png', '2358.png']
# particolare = ['6062.png', '7425.png', '5569.png', '1733.png', '14269.png', '7626.png', '15413.png', '2372.png', '3911.png']

# 6000
# bad_segmented = ['1679.png', '2364.png', '2827.png', '6466.png', '3495.png', '3324.png', '7624.png', '12418.png', '1527.png', '4305.png', '2012.png', '7426.png', '3898.png', '7340.png', '14525.png', '2204.png', '2210.png', '7369.png', '1718.png', '11474.png', '1724.png', '3655.png', '6088.png', '2588.png', '3682.png', '2013.png', '10540.png', '3457.png', '2985.png', '3494.png']
# to_rotate_or_flip = ['12593.png', '6328.png', '1447.png', '11717.png', '8463.png', '3481.png', '13074.png', '3456.png', '2006.png', '2774.png', '10780.png', '1719.png', '14242.png', '8728.png', '4112.png', '1730.png', '8927.png', '1917.png', '11892.png', '14726.png']
# particolare = ['13458.png', '14860.png', '11663.png', '8067.png', '3325.png']

# 6500
# bad_segmented = ['7779.png', '9026.png', '2173.png', '2365.png', '8890.png', '10636.png', '1650.png', '3053.png', '4043.png', '15019.png', '2340.png', '3076.png', '7576.png', '1675.png', '11041.png', '2022.png', '2585.png']
# to_rotate_or_flip = ['15216.png', '5185.png', '4072.png', '1877.png', '1863.png', '14493.png', '4080.png', '12004.png', '1885.png', '14475.png', '5349.png', '1891.png', '1488.png', '7990.png', '13118.png', '3274.png', '1477.png', '11727.png', '4269.png', '2787.png', '14071.png', '10798.png']
# particolare = ['5016.png', '10385.png', '11500.png', '13326.png', '7562.png', '13130.png', '14918.png', '12819.png']

# #7000
# bad_segmented = ['1729.png', '7358.png', '6720.png', '3880.png', '1701.png', '1700.png', '1502.png', '3473.png', '1516.png', '10558.png', '7007.png', '1476.png', '2143.png', '3249.png', '14689.png', '1489.png', '12987.png', '3711.png', '2341.png', '1648.png', '1884.png']
# to_rotate_or_flip = ['4137.png', '1714.png', '6053.png', '5228.png', '1927.png', '2584.png', '3301.png', '5770.png', '2962.png', '2786.png', '6286.png', '4240.png', '2157.png', '12207.png', '2180.png', '2194.png', '5374.png', '15024.png', '9228.png', '8678.png', '1879.png']
# particolare = ['14265.png', '2544.png', '6709.png', '7826.png', '10362.png', '9943.png', '8108.png', '6643.png']

# 7500
# bad_segmented = ['3921.png', '4068.png', '1662.png', '4295.png', '8493.png', '4256.png', '10214.png', '15378.png', '12417.png', '8901.png', '4875.png', '3897.png', '3100.png', '1717.png', '3672.png', '2222.png', '12600.png', '4121.png', '3896.png', '6905.png', '1918.png', '11137.png', '13708.png', '6534.png', '2034.png']
# to_rotate_or_flip = ['8938.png', '1886.png', '3061.png', '13443.png', '1448.png', '9000.png', '15542.png', '2948.png', '7830.png', '4322.png', '1500.png', '3303.png', '1919.png', '12832.png', '4120.png', '1702.png', '4108.png', '1716.png', '3114.png', '4135.png', '8082.png', '3855.png', '1515.png', '1501.png', '1529.png']
# particolare = ['8451.png', '11917.png', '11863.png', '10758.png', '8055.png', '5997.png']

# 8000
# bad_segmented = ['11057.png', '11527.png', '14463.png', '1844.png', '3919.png', '8680.png', '8858.png', '3925.png', '6691.png', '11537.png', '2352.png', '12002.png', '2178.png', '3514.png', '3528.png', '9159.png', '1511.png', '14711.png', '4327.png', '3448.png', '4333.png']
# to_rotate_or_flip = ['2838.png', '11043.png', '3289.png', '8647.png', '1663.png', '1677.png', '4041.png', '9571.png', '13495.png', '1850.png', '13491.png', '1673.png', '9788.png', '1471.png', '8441.png', '6322.png', '2622.png', '1459.png', '7809.png', '2965.png', '13095.png', '11682.png', '1505.png', '3460.png', '2030.png']
# particolare = ['4243.png', '5834.png', '13481.png', '3935.png', '7570.png', '6450.png']

# 8500
# bad_segmented = ['3674.png', '7389.png', '8910.png', '3110.png', '1713.png', '3104.png', '2232.png', '2226.png', '3886.png', '6041.png', '2541.png', '3688.png', '15181.png', '3449.png', '4326.png', '3475.png', '7029.png', '1458.png', '2623.png', '4285.png', '2192.png', '10628.png', '14328.png', '1672.png', '13484.png']
# to_rotate_or_flip = ['7956.png', '1934.png', '15157.png', '1707.png', '14261.png', '10760.png', '3887.png', '9366.png', '14512.png', '3663.png', '6900.png', '4865.png', '2743.png', '2757.png', '3515.png', '13645.png', '9038.png', '11085.png', '1896.png', '12759.png', '3703.png']
# particolare = ['5211.png', 9401.png', '13719.png', '9762.png', '15234.png', '12003.png', '4722.png', '7559.png', '8124.png', '5366.png', '9548.png']

# 9000
# bad_segmented = ['4085.png', '3715.png', '1664.png', '1880.png', '13690.png', '4293.png', '12559.png', '2153.png', '12217.png', '3477.png', '2027.png', '2755.png', '10574.png', '1923.png', '6094.png', '8907.png', '5210.png', '3675.png', '3649.png', '1922.png', '6917.png', '2754.png', '3338.png']
# to_rotate_or_flip = ['8587.png', 3926.png', '2379.png', '12997.png', '1658.png', '1499.png', '7942.png', '3517.png', '1472.png', '14672.png', '6321.png', '1512.png', '13055.png', '3339.png', '12149.png', '1704.png', '2231.png', '4133.png', '1711.png', '14539.png', '3847.png', '1936.png', '10788.png', '2032.png']
# particolare = ['10658.png', '6490.png', '6484.png', '1466.png', '9833.png', '7822.png', '15430.png', '4126.png', '3884.png', '4325.png', '11119.png']

# 9500
# bad_segmented = ['3476.png', '3462.png', '12570.png', '7994.png', '4279.png', '14868.png', '4286.png', '3728.png', '1659.png', '4090.png', '6877.png', '9289.png', '2337.png', '3983.png', '3767.png', '13184.png', '6353.png', '11988.png', '2096.png', '2733.png']
# to_rotate_or_flip = ['1507.png', '8245.png', '7758.png', '9013.png', '1473.png', '2813.png', '1895.png', '4047.png', '13487.png', '12941.png', '7515.png', '1399.png', '4222.png', '10300.png', '7717.png', '2082.png', '2727.png']
# particolare = ['5507.png', '5934.png', 13726.png', '14049.png', '5946.png', '11910.png', '8872.png', '8140.png', '3388.png', '14012.png']

# 10k
# bad_segmented = ['3363.png', '3820.png', '4154.png', '4168.png', '5505.png', '3174.png', '1763.png', '3606.png', '2518.png', '1987.png', '4627.png', '4141.png', '11382.png', '3821.png', '9883.png', '11023.png', '3564.png', '8431.png', '4237.png', '8419.png', '2861.png', '3766.png']
# to_rotate_or_flip = ['14950.png', '12305.png', '3377.png', '12339.png', '1945.png', '1979.png', '1992.png', '2256.png', '11369.png', '2242.png', '12853.png', '1944.png', '13768.png', '2083.png', '4579.png', '14167.png', '9707.png', '1398.png', '7266.png', '9539.png', '2322.png']
# particolare = ['7884.png', 14011.png', 2687.png', 8785.png', '7449.png', '11619.png', '9908.png', '8633.png', '11235.png']

# 10500
# bad_segmented = ['3969.png', '1830.png', '11579.png', '3764.png', '1601.png', '15051.png', '2650.png', '2136.png', '2056.png', '1563.png', '3837.png', '13540.png', '5513.png', '3611.png', '1748.png', '5248.png', '10074.png', '14212.png']
# to_rotate_or_flip = ['15079.png', '1629.png', '3994.png', '7927.png', '12299.png', '4235.png', '1588.png', '3406.png', '5077.png', '1952.png', '1985.png', '1761.png', '1775.png', '3163.png', '6769.png', '10048.png']
# particolare = ['7112.png']

# 11000
# bad_segmented = ['8977.png', '6796.png', '3836.png', '4354.png', '14010.png', '15497.png', '3573.png', '8368.png', '1600.png', '8142.png', '3771.png', '1614.png', '3759.png', '3956.png', '3007.png', '3013.png', '6382.png', '9072.png', '2133.png', '2655.png']
# to_rotate_or_flip = ['1576.png', '15454.png', '2725.png', '2057.png', '2731.png', '4340.png', '13780.png', '5089.png', '13145.png', '1402.png', '4208.png', '2137.png', '3229.png', '13186.png', '4585.png', '2309.png', '10672.png', '1833.png', '12920.png', '7507.png', '4740.png', '8387.png', '2682.png', '3588.png', '1406.png', '2669.png', '14160.png']
# particolare = ['2296.png', '5937.png', '11222.png', '3003.png', '9258.png', '9516.png', '7249.png', '9728.png']

# 11500
# bad_segmented = ['13784.png', '6209.png', '10500.png', '1764.png', '8740.png', '5516.png', '1995.png', '3628.png', '5259.png', '1956.png', '3833.png', '13593.png', '5729.png', '6546.png', '5073.png', '2085.png']
# to_rotate_or_flip = ['2084.png', '9660.png', '2053.png', '3359.png', '1943.png', '4185.png', '13551.png', '5270.png', '1980.png', '12672.png', '10071.png', '7300.png', '11421.png', '3166.png', '3614.png', '6234.png', '1407.png']
# particolare = ['13974.png', '8218.png', '9489.png', '8967.png', '15479.png', '8225.png', '12074.png']

# 12000
# bad_segmented = ['5305.png', '13424.png', '1605.png', '3012.png', '10852.png', '8192.png', '3951.png', '1607.png', '13426.png', '3004.png', '6632.png', '1388.png', '7921.png', '3574.png', '3212.png', '4569.png', '13156.png', '2130.png', '5893.png', '2642.png', '2939.png', '2087.png', '2078.png', '1559.png', '1798.png', '3616.png', '4179.png', '4623.png']
# to_rotate_or_flip = ['2668.png', '3403.png', '15041.png', '7510.png', '2454.png', '6626.png', '2440.png', '8582.png', '1565.png', '2736.png', '3825.png', '1940.png', '1767.png', '1997.png', '12664.png']
# particolare = ['5306.png', 3213.png', 15069.png', '9271.png', '15321.png', '4796.png', '6154.png', '4031.png', '14836.png', '11769.png', '7048.png', '9893.png', '8233.png', '10067.png', '10715.png']

# 12500
# bad_segmented = ['10927.png', '12301.png', '13989.png', '2086.png', '7049.png', '11740.png', '3207.png', '1404.png', '2327.png', '5448.png', '9266.png', '8150.png', '4018.png', '1835.png', '3978.png']
# to_rotate_or_flip = ['1415.png', '6172.png', '11422.png', '1941.png', '1955.png', '2045.png', '3415.png', '5919.png', '2858.png', '3987.png', '12077.png', '1612.png', '2482.png', '4971.png']
# particolare = ['3156.png', '1766.png', '6784.png', '1969.png', '11387.png', '9110.png', '7840.png', '2938.png', '14604.png', '9919.png']


# images_path = "/Users/alex/Desktop/reduced_handset/training/"
# images = os.listdir(images_path)
# bad_segmented = []
# to_remove = []
# to_rotate_or_flip = []
# particolare = []
# threshold = 9847
# range = 0
# i = 0
# print(len(images) - threshold)
# for elem in images:
#     # if (elem in removexzd_train1):# and not(elem in removed_val1):
#     i += 1
#     # if threshold - range <= i < threshold:
#     if i == threshold:
#         # print elem
#         img = cv2.imread(images_path + elem)
#         while True:
#             cv2.imshow(str(i) + "  " + elem, img)
#             k = cv2.waitKey(33)
#             if k == 27:
#                 break
#             elif k == -1:
#                 continue
#             elif k == 13:  # enter
#                 bad_segmented.append(elem)
#                 break
#             elif k == 114:  # r
#                 to_remove.append(elem)
#                 break
#             elif k == 122:  # z
#                 to_rotate_or_flip.append(elem)
#                 break
#             elif k == 112:  # p
#                 particolare.append(elem)
#                 break
#             else:
#                 print(k)
#     elif i < threshold - range:
#         continue
#     else:
#         break
# print("bad_segmented =", bad_segmented)
# print("to_rotate_or_flip =", to_rotate_or_flip)
# print("particolare =", particolare)
# print("to_remove =", to_remove)

# import cv2
# import numpy as np
#
# folder = "/Users/alex/Desktop/handset/validation2/"
# file = "9760.png"
#
# img = cv2.imread(folder + file, cv2.IMREAD_GRAYSCALE)
# contours, hierarchy = cv2.findContours(img, 1, 2)
# cnt = contours[0]
#
# rect = cv2.minAreaRect(cnt)
# box = cv2.boxPoints(rect)
# box = np.int0(box)
# newImg = cv2.drawContours(img,[box],0,(0,255,255),2)
# cv2.drawContours(img, [box], -1, (255, 255, 0), 1)
#
# while (1):
#     cv2.imshow("box", img)
#     k = cv2.waitKey(33)
#     if k == 27:
#         break
#     elif k == -1:
#         continue
#     else:
#         print k

images_path = "/Users/alex/Desktop/test.png"
img = cv2.imread(images_path, 0)
clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(133, 124))
img1 = clahe.apply(img)
while True:
    cv2.imshow("box", img1)
    k = cv2.waitKey(33)
    if k == 27:
        break

# from sklearn.model_selection import KFold
# import numpy as np
#
# X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
# y = np.array([1, 2, 3, 4])
# kf = KFold(n_splits=5, shuffle=True)
#
# for train_index, test_index in kf.split(X):
#     print("TRAIN:", train_index, "TEST:", test_index)
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]

