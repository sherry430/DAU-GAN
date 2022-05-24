import glob
import sys
from random import seed, shuffle

import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial
from PIL import Image
from sklearn.metrics import (confusion_matrix, precision_recall_curve)


def load_data(file_path):
    return glob.glob(file_path)


def read_image(fn, imageSize):
    im = Image.open(fn)
    im = im.resize((imageSize, imageSize), Image.BILINEAR)
    # arr = np.array(im)/1
    arr = np.asarray(im).astype(np.float32)
    if len(arr.shape) == 2:
        arr = np.reshape(arr, (imageSize, imageSize, 1))
    if arr.shape[2] == 3:
        edge = arr[..., 0] < 210.0
        mean = np.mean(arr[...][edge], axis=0)
        std = np.std(arr[...][edge], axis=0)
        return (arr - mean) / std
    else:
        if np.max(arr) > 1:
            arr = arr / 255
            assert(np.min(arr) == 0 and np.max(arr) == 1)
        return arr


def minibatch(dataAFile, dataBFile, batchSize, imageSize, test=False):
    assert len(dataAFile) == len(dataBFile)
    length = len(dataAFile)
    epoch = i = 0
    tmpSize = None
    while True:
        size = tmpSize if tmpSize else batchSize
        if i + size > length:
            # set random.seed to maintain relation between dataA/dataB
            if not test:
                seed(100)
                shuffle(dataAFile)
                seed(100)
                shuffle(dataBFile)
            i = 0
            epoch += 1
        dataA = []
        dataB = []

        for j in range(i, i + size):
            imgA = read_image(dataAFile[j], imageSize)
            imgB = read_image(dataBFile[j], imageSize)
            dataA.append(imgA)
            dataB.append(imgB)
        dataA = np.float32(dataA)
        dataB = np.float32(dataB)
        i += size
        tmpSize = yield epoch, dataA, dataB


def input2discriminator(real_A, real_B, fake_B):
    real = np.concatenate((real_A, real_B), axis=3)
    fake = np.concatenate((real_A, fake_B), axis=3)

    d_x_batch = np.concatenate((real, fake), axis=0)
    d_y_batch = np.ones((d_x_batch.shape[0], 1))
    d_y_batch[real_A.shape[0]:, ...] = 0

    return d_x_batch, d_y_batch


def input2gan(real_A, real_B):
    g_x_batch = [real_A, real_B]
    g_y_batch = np.ones((real_B.shape[0], 1))
    return g_x_batch, g_y_batch


def best_f1_threshold(precision, recall, thresholds):
    best_f1 = -1
    for index in range(len(precision)):
        curr_f1 = 2.*precision[index]*recall[index] / \
            (precision[index]+recall[index])
        if best_f1 < curr_f1:
            best_f1 = curr_f1
            best_threshold = thresholds[index]

    return best_f1, best_threshold


def threshold_by_f1(true_vessels, generated, flatten=True, f1_score=False):
    precision, recall, thresholds = precision_recall_curve(true_vessels.flatten().flatten(),
                                                           generated.flatten().flatten(),
                                                           pos_label=1)
    best_f1, best_threshold = best_f1_threshold(precision, recall, thresholds)

    pred_vessels_bin = np.zeros(generated.shape)
    pred_vessels_bin[generated >= best_threshold] = 1

    if flatten:
        if f1_score:
            return pred_vessels_bin.flatten(), best_f1, best_threshold
        else:
            return pred_vessels_bin.flatten()
    else:
        if f1_score:
            return pred_vessels_bin, best_f1, best_threshold
        else:
            return pred_vessels_bin


def misc_measures(true_vessels, pred_vessels):
    thresholded_vessel_arr, f1_score, best_threshold = threshold_by_f1(true_vessels,
                                                                       pred_vessels,
                                                                       f1_score=True)
    true_vessel_arr = true_vessels.flatten()

    cm = confusion_matrix(true_vessel_arr, thresholded_vessel_arr)
    acc = 1.*(cm[0, 0]+cm[1, 1])/np.sum(cm)
    sensitivity = 1.*cm[1, 1]/(cm[1, 0]+cm[1, 1])
    specificity = 1.*cm[0, 0]/(cm[0, 1]+cm[0, 0])
    return f1_score, acc, sensitivity, specificity, cm, best_threshold


def getDSC(testImage, resultImage):
    """Compute the Dice Similarity Coefficient."""
    testImage[testImage >= 1] = 1
    resultImage[resultImage >= 1] = 1

    testArray = testImage.flatten()
    resultArray = resultImage.flatten()

    return 1.0 - scipy.spatial.distance.dice(testArray, resultArray)


def print_metrics(itr, **kargs):
    print("*** Round {}  ====> ".format(itr)),
    for name, value in kargs.items():
        print("{} : {}, ".format(name, value)),
    print("")

    sys.stdout.flush()


def plot_metrics(epoch, loss, f1, savePath):
    x = range(0, epoch+1, 1)
    plt.plot(x, loss, 'b', label='loss')
    plt.plot(x, f1, 'r', label='f1')
    plt.legend()
    plt.grid(True)
    plt.xlabel('epoch')
    plt.savefig(savePath)
    plt.close()


if __name__ == "__main__":
    dir = './data/liver720_1/train/label/02.png'
    img = read_image(dir, 720)
