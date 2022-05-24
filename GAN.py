import argparse
import os
import time

import keras.backend as K
import numpy as np
from PIL import Image

from model import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--ratio_gan2seg',
                    type=int,
                    help="ratio of gan loss to seg loss",
                    default=10)
parser.add_argument('--gpu_index', type=str, help="gpu index", default='0')
parser.add_argument('--batch_size', type=int, help="batch size", default=2)
parser.add_argument('--lr',
                    type=float,
                    help="initial learning rate",
                    default=1e-4)
parser.add_argument('--imageSize', type=int, help="imageSize", default=720)
parser.add_argument('--dataset',
                    type=str,
                    help="type of dataset",
                    default='liver720_1')
parser.add_argument('--withAugment',
                    action='store_true',
                    help="whether the dataset has been augmented")
FLAGS, _ = parser.parse_known_args()


os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_index
alpha_recip = 1. / FLAGS.ratio_gan2seg if FLAGS.ratio_gan2seg > 0 else 0
imageSize = FLAGS.imageSize
batchSize = FLAGS.batch_size
dataset = FLAGS.dataset
init_lr = FLAGS.lr
withAugment = FLAGS.withAugment

print(withAugment)
bestModelSave = 0.85
epochs = 100
patience = 20
n_filters_d = 64
n_filters_g = 32
earlystopCount = 0
lrAdjustCount = 0
min_loss = float("inf")
overfitting_patience = 3
resumeTrain = False
origin_vessel = 'vessel'    # the input dir of vessel image
label = 'label' # the input dir of label image

if withAugment == True:
    # dataset after augmentation used to train
    train_data_dirs = './augment/' + dataset + '/train/' + origin_vessel + '/*.png'
    train_label_dirs = './augment/' + dataset + '/train/' + label + '/*.png'
else:
    # dataset before augmentation used to train
    train_data_dirs = './data/' + dataset + '/train/' + origin_vessel + '/*.png'
    train_label_dirs = './data/' + dataset + '/train/' + label + '/*.png'

# dataset used to test
test_data_dir = './data/' + dataset + '/test/' + origin_vessel + '/*.png'
test_label_dir = './data/' + dataset + '/test/' + label + '/*.png'

# output dirs
epoch_result_save = './data/' + dataset + '/res/' + str(imageSize) + '/vessel'
# output bw dirs
bw_result_save = './data/' + dataset + '/res/' + str(imageSize) + '/bw'
model_path = './data/' + dataset + '/model/'
metrics_path = './data/' + dataset + '/res/' + str(imageSize) + '/metric.png'

def mkdir(path):
    os.makedirs(path, exist_ok=True)

mkdir(epoch_result_save)
mkdir(bw_result_save)
mkdir(model_path)

K.set_image_data_format('channels_last')

netG = generator_da_unet(imageSize, n_filters_g)
netG.summary()

netD = discriminator(imageSize, n_filters_d, init_lr)
netD.summary()

gan = GAN(netG, netD, imageSize, alpha_recip, init_lr)
gan.summary()

# load Data
train_data = load_data(train_data_dirs)
train_label = load_data(train_label_dirs)

test_data = load_data(test_data_dir)
test_label = load_data(test_label_dir)

test_filename = []
for test_file in test_data:
    temp_string = test_file.split('/')
    test_filename.append(temp_string[-1][:-4])

print(
    'Train Data: %d its,  %d its \n Test Data: %d its,  %d its'
    % (len(train_data), len(train_label),
       len(test_data), len(test_label)))
assert len(train_data) and len(train_label) and len(test_data) and len(test_label)

# train Model
if resumeTrain:
    print('Load model files to continue training...')
    modelGPath = model_path + '/modelG_' + origin_vessel + 'db.h5'
    modelDPath = model_path + '/modelD_' + origin_vessel + 'db.h5'
    netG.load_weights(modelGPath)
    netD.load_weights(modelDPath)

t0 = time.time()

gen_iterations = 1
curEpoch = 0

train_iters = len(train_data) // batchSize

train_batch = minibatch(train_data,
                        train_label,
                        batchSize=batchSize,
                        imageSize=imageSize)
test_batch = minibatch(test_data,
                       test_label,
                       batchSize=len(test_data),
                       imageSize=imageSize,
                       test=True)
loss_D = []
acc_D = []
loss_gan = []
acc_gan = []
f1_mean_test = []
loss_mean_test = []

while curEpoch < epochs:
    modelGPath = model_path + '/modelG_' + origin_vessel + 'db.h5'
    modelDPath = model_path + '/modelD_' + origin_vessel + 'db.h5'
    if earlystopCount > patience:
        print("early stop in epoch : ", curEpoch, ", patience : ", patience)
        break

    # train
    curEpoch, trainA, trainB = next(train_batch)
    # train D
    netD.trainable = True
    d_x_batch, d_y_batch = input2discriminator(
        trainA, trainB, netG.predict(trainA, batch_size=batchSize))
    d_loss, d_acc = netD.train_on_batch(d_x_batch, d_y_batch)
    loss_D.append(d_loss)
    acc_D.append(d_acc)

    # train G
    netD.trainable = False
    g_x_batch, g_y_batch = input2gan(trainA, trainB)
    g_loss, g_acc = gan.train_on_batch(g_x_batch, g_y_batch)
    loss_gan.append(g_loss)
    acc_gan.append(g_acc)

    if gen_iterations % train_iters == 0:
        print_metrics(curEpoch,
                      loss=np.mean(loss_D),
                      acc=np.mean(acc_D),
                      type='DIS')
        print_metrics(curEpoch,
                      loss=np.mean(loss_gan),
                      acc=np.mean(acc_gan),
                      type='GAN')    
        # test
        _, testA, testB = next(test_batch)
        generated = netG.predict(testA, batch_size=len(test_data))
        generated = np.squeeze(generated, axis=3)

        g_x_batch_test, g_y_batch_test = input2gan(testA, testB)
        gan_loss, gan_acc = gan.evaluate(g_x_batch_test, g_y_batch_test,verbose=0)
        loss_mean_test.append(gan_loss)
        if loss_mean_test[curEpoch] < min_loss:
            earlystopCount = 0
            lrAdjustCount = 0
            min_loss = loss_mean_test[curEpoch]
            print("update min_loss: ", min_loss, " early_stop_count", earlystopCount)
        else:
            earlystopCount += 1
            lrAdjustCount += 1
            print("current min_loss: ", min_loss, " early_stop_count", earlystopCount)

        f1_score, acc, sensitivity, specificity, cm, best_threshold = misc_measures(testB, generated)
        print_metrics(curEpoch, f1_score=f1_score, acc=acc, sensitivity=sensitivity, specificity=specificity, best_threshold=best_threshold, type='TESTING_set', gan_loss=gan_loss)

        # print test images
        f1 = []
        for index in range(len(test_data)):
            Image.fromarray(
                (generated[index, :, :] * 255).astype(np.uint8)).save(
                    os.path.join(
                        epoch_result_save,
                        str(curEpoch) + "_" + test_filename[index] +
                        ".png"))
            bw_image = generated[index, :, :]
            bw_image = np.where(bw_image >= best_threshold, 255, bw_image)
            bw_image = np.where(bw_image < best_threshold, 0, bw_image)
            Image.fromarray(
                bw_image.astype(np.uint8)).save(
                    os.path.join(
                        bw_result_save,
                        str(curEpoch) + "_" + test_filename[index] +
                        ".png"))
            f1.append(getDSC(testB[index, :, :],bw_image))
        print_metrics(curEpoch, f1=f1, f1_mean = np.mean(f1))
        f1_mean_test.append(np.mean(f1))

        # save currently best model
        if np.mean(f1) > bestModelSave:
            bestModelSave = np.mean(f1)
            print("currently best model's f1 score : ", bestModelSave)
            netD.save(modelDPath)
            netG.save(modelGPath)
        plot_metrics(curEpoch, loss_mean_test, f1_mean_test, metrics_path)
        now_lr = K.get_value(gan.optimizer.lr)
        print('current learning rate : ', now_lr)
        # update learning rate
        if lrAdjustCount >= overfitting_patience:
            new_lr = now_lr*0.5
            K.set_value(gan.optimizer.lr, new_lr)
            print('update learning rate : ', new_lr)
            lrAdjustCount = 0
    gen_iterations += 1
