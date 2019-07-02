import glob
import os
import re

import cv2
import numpy as np
import scipy
import tensorflow as tf
import tensorflow.python.keras as keras

import config
from threadsafe_iterator import threadsafe_generator


def _log10(x):
    return tf.log(x) / tf.log(tf.constant(10, dtype=x.dtype))

def PSNR(y_true, y_pred):
    # Peak Signal to Noise Ratio
    max_pixel = 1.
    return 10. * _log10((max_pixel ** 2) / (keras.backend.mean(keras.backend.square(y_pred - y_true))))

def load_images(directory):
    images = []

    for root, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if re.search('\.(jpg|jpeg|png|bmp|tiff)$', filename):
                
                filepath = os.path.join(root, filename)
                image = cv2.imread(filepath)

                images.append(image)
    
    images = np.array(images)
    shape = np.append(images.shape[0:3], 1)
    images = np.resize(images, (shape))

    return images

def get_images_list(data_path, scales=[2, 3, 4]):
    l = glob.glob(os.path.join(data_path, ''))
    l = [f for f in l if re.search('^\d+.mat$', os.path.basename(f))]

    train_list = []

    for f in l:
        if os.path.exists(f):
            for scale in scales:
                string_scale = '_' + str(scale) + '.mat'

                if os.path.exists(f[:-4] + string_scale):
                    train_list.append(f, f[:-4] + string_scale)
    
    return train_list

def _get_image_batch(train_list, offset):
    target_list = train_list[offset : offset + config.BATCH_SIZE]
    input_list = []
    gt_list = []
    cbcr_list = []

    for pair in target_list:
        input_image = scipy.io.loadmat(pair[1])['patch']
        gt_image = scipy.io.loadmat(pair[0])['patch']
        
        input_list.append(input_image)
        gt_list.append(gt_image)
    
    input_list = np.array(input_list)
    input_list.resize([config.BATCH_SIZE, config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], 1])

    gt_list = np.array(gt_list)
    gt_list.resize([config.BATCH_SIZE, config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], 1])

    return input_list, gt_list

@threadsafe_generator
def image_generator(target_list):
    while True:
        for step in range(len(target_list) // config.BATCH_SIZE):
            offset = step * config.BATCH_SIZE

            batch_x, batch_y = _get_image_batch(target_list, offset)

            yield(batch_x, batch_y)