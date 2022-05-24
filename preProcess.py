import os
from PIL import Image, ImageEnhance
# from keras.preprocessing.image import Iterator
from scipy.ndimage import rotate
# from skimage import filters
# from sklearn.metrics import auc
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
# from skimage import measure
# from random import randint, shuffle, seed
# import matplotlib.pyplot as plt
import numpy as np
import random
import shutil

def mkdir(path):
    os.makedirs(path, exist_ok=True)

def random_perturbation(imgs):
    im = Image.fromarray(imgs.astype(np.uint8))
    en = ImageEnhance.Brightness(im)
    im = en.enhance(random.uniform(0.8,1.2))
    en = ImageEnhance.Color(im)
    im = en.enhance(random.uniform(0.8,1.2))
    en = ImageEnhance.Contrast(im)
    im = en.enhance(random.uniform(0.8,1.2))
    imgs = np.asarray(im).astype(np.float32)
    return imgs 


def augment(data_dir, label_dir):
    origin_index = 1
    label_index = 1
    origin = Image.open(data_dir)
    origin_arr = np.asarray(origin).astype(np.float32)

    label = Image.open(label_dir)
    label_arr = np.asarray(label).astype(np.float32)

    data_dir = data_dir.replace('data','augment')
    label_dir = label_dir.replace('data','augment')

    flipped_origin = origin_arr[:,::-1,:]    # flipped imgs
    flipped_label = label_arr[:,::-1]
    flipped_label_img =  Image.fromarray(flipped_label)
    
    if np.max(label_arr == 1.0):
        label_arr *= 255

    origin.save(data_dir)
    Image.fromarray(label_arr.astype(np.uint8)).save(label_dir)
    save_origin_dir = data_dir[:-4] + '_0.png'
    Image.fromarray(flipped_origin.astype(np.uint8)).save(save_origin_dir)
    save_label_dir = label_dir[:-4] + '_0.png'
    Image.fromarray(flipped_label.astype(np.uint8)).save(save_label_dir)

    for angle in range(0,360,4):  # rotated imgs 3~360  (3,360,3)
        save_origin_dir = data_dir[:-4] + '_' + str(origin_index) + '.png'
        rotate_origin = random_perturbation(rotate(origin_arr, angle, axes=(0, 1), reshape=False))
        Image.fromarray(rotate_origin.astype(np.uint8)).save(save_origin_dir)
        origin_index += 1
        save_origin_dir = data_dir[:-4] + '_' + str(origin_index) + '.png'
        rotate_origin_flip = random_perturbation(rotate(flipped_origin, angle, axes=(0, 1), reshape=False))
        Image.fromarray(rotate_origin_flip.astype(np.uint8)).save(save_origin_dir)
        origin_index += 1

        save_label_dir = label_dir[:-4] + '_' + str(label_index) + '.png'
        rotate_label = np.asarray(label.rotate(angle)) * 255
        Image.fromarray(rotate_label.astype(np.uint8)).save(save_label_dir)
        label_index += 1
        save_label_dir = label_dir[:-4] + '_' + str(label_index) + '.png'
        rotate_label_flip = np.asarray(flipped_label_img.rotate(angle)) * 255
        Image.fromarray(rotate_label_flip.astype(np.uint8)).save(save_label_dir)
        label_index += 1

if __name__ == "__main__":
    train_data_dirs = './data/liver720_1/train/vessel/'
    train_label_dirs = './data/liver720_1/train/label/'
    test_data_dirs = './data/liver720_1/test/vessel/'
    test_label_dirs = './data/liver720_1/test/label/'
    
    augmented_train_data_dirs = train_data_dirs.replace("data","augment")
    augmented_train_label_dirs = train_label_dirs.replace("data","augment")
    augmented_test_data_dirs = test_data_dirs.replace("data","augment")
    augmented_test_label_dirs = test_label_dirs.replace("data","augment")

    mkdir(augmented_train_data_dirs)
    mkdir(augmented_train_label_dirs)
    mkdir(augmented_test_data_dirs)
    mkdir(augmented_test_label_dirs)
    print("processing the 1 flod data augmentation.")
    for _, filename in enumerate(os.listdir(train_data_dirs)):
        train_data_dir = train_data_dirs + filename
        train_label_dir = train_label_dirs + filename
        augment(train_data_dir, train_label_dir)

    for _, filename in enumerate(os.listdir(test_data_dirs)):
        test_data_dir = test_data_dirs + filename
        test_label_dir = test_label_dirs + filename
        augment(test_data_dir, test_label_dir)

    pathlist = os.listdir(augmented_train_data_dirs) + os.listdir(augmented_test_data_dirs)
    for i in range(2,5):
        mkdir('augment/liver720_' + str(i) + '/train/vessel')
        mkdir('augment/liver720_' + str(i) + '/test/vessel')
        mkdir('augment/liver720_' + str(i) + '/train/label')
        mkdir('augment/liver720_' + str(i) + '/test/label')
    for i in range(2,5):
        print("processing the ", i, " flod data augmentation.")
        for _,filename in enumerate(pathlist):
            if not ('_' in filename):
                filename_list = filename.split('.')
            else:
                filename_list = filename.split('_')
            if int(filename_list[0]) % 4 == i % 4 :
                savepath_vessel = 'augment/liver720_' + str(i) + '/test/vessel/' + filename
                savepath_label = 'augment/liver720_' + str(i) + '/test/label/' + filename
            else:
                savepath_vessel = 'augment/liver720_' + str(i) + '/train/vessel/' + filename    
                savepath_label = 'augment/liver720_' + str(i) + '/train/label/' + filename  
            if int(filename_list[0]) % 4 == 1:
                sourcePath_vessel = augmented_test_data_dirs + filename
                sourcePath_label = augmented_test_label_dirs + filename
            else:
                sourcePath_vessel = augmented_train_data_dirs + filename
                sourcePath_label = augmented_train_label_dirs + filename
            shutil.copy(sourcePath_vessel, savepath_vessel)
            shutil.copy(sourcePath_label, savepath_label)