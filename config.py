"""
File: config.py
Created by Andrew Ingson (aings1@umbc.edu)
Date: 4/19/2021
CMSC 4-- (CLASS NAME)

Helper functions
"""

import tensorflow as tf

import models.facenet_resnet_pretrained as facenet

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

inp_size = (160, 160)
