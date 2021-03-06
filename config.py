"""
File: config.py
Created by Andrew Ingson (aings1@umbc.edu) & Chase Carbaugh (aings1@umbc.edu & chasec1@umbc.edu)
Date: 4/19/2021
CMSC 491 Special Topics - Computer Vision

Helper functions (get used by classifier and primary so they are in this file so we are DRY)
"""

import logging
import os

import mtcnn_cv2
import tensorflow as tf

import models.facenet_resnet_pretrained as facenet

# what size image should the preprocessing create
inp_size = (160, 160)
# start mtcnn backend
detector = mtcnn_cv2.MTCNN()
# should tf use the gpu
use_gpu = False
# should inputs be resized
resize_inputs = True
# should frame be resized
resize_frame = False
# what size should the inputs be resized too
resize_target = 250, 250
# what classifier do we want to use
classifier_target = "output/my_classifier.pkl"
# file to hold classifier history in
classifier_hist_path = "hist.pkl"
classifier_hash_path = "hash.pkl"
# how many faces should we store for a class
face_limit = 100
# how often to save faces
face_rate = 3
# how many seconds of inactivity do we wait before re-classification
inactivity_thresh = 1.5
# where are inputs stored
input_directory = 'input'

log = logging.getLogger('tensorflow')
log.setLevel(logging.ERROR)

if not use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6144)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

main_model = facenet.inception_resnet_v2_pre()
main_model.load_weights("models/facenet_pretrained.h5")
# applications.InceptionResNetV2(include_top=True, weights="imagenet", pooling='avg', classifier_activation='softmax')
# applications.InceptionResNetV2(include_top=False, weights="imagenet", input_shape=(160, 160, 3), pooling='avg')
