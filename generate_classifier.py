"""
File: generate_classifier.py
Created by Andrew Ingson & Chase Carbaugh (aings1@umbc.edu & chasec1@umbc.edu)
Date: 4/19/2021
CMSC 491 Special Topics - Computer Vision

"""

import argparse
import os
import pickle

import cv2
import mtcnn
import numpy as np

from config import *


def main():
    parser = argparse.ArgumentParser(description='Classifier Generator')
    parser.add_argument("-v", "--verbosity", type=int, help="increase output verbosity", default=0)
    args = parser.parse_args()

    if args.verbosity > 0:
        verbose = True
    else:
        verbose = False
    input_directory = 'input'
    # hold references to completed classes
    completed_classes = dict()
    # start mtcnn backend
    detector = mtcnn.MTCNN()

    for possible_class in os.listdir(input_directory):
        if "README" not in possible_class:
            combined_class_dir = os.path.join(input_directory, possible_class)
            print("\n\nTraining on", len(os.listdir(combined_class_dir)), "for", possible_class)
            # set up varibles for inner loop
            prediction_results = []
            img_count = 0
            for source_image in os.listdir(combined_class_dir):
                # skip readme files since they are not images
                if "README" not in source_image:
                    print(str(img_count) + ":", source_image)
                    img_count += 1
                    img_path = os.path.join(combined_class_dir, source_image)
                    img = cv2.imread(img_path)

                    # get face location from mtcnn
                    results = detector.detect_faces(img)
                    # if an image is found
                    if len(results) > 0:
                        # trim the sample image to be only the target face
                        bounding_box = results[0]['box']
                        bounding_box[0], bounding_box[1] = abs(bounding_box[0]), abs(bounding_box[1])
                        face = img[bounding_box[1]:bounding_box[1] + bounding_box[3],
                               bounding_box[0]:bounding_box[0] + bounding_box[2]]
                        # keep track of image manipulation in-case we are in verbose mode
                        stage1 = face

                        # normalize the face image from uint8 to float64
                        face = (face - face.mean()) / face.std()
                        stage2 = face
                        # resize the face image to the input tensor size for the model
                        face = cv2.resize(face, inp_size)

                        # create training live feed if we are in verbose mode
                        if verbose:
                            # display float64 image
                            scale_percent = 200  # percent of original size
                            width = int(face.shape[1] * scale_percent / 100)
                            height = int(face.shape[0] * scale_percent / 100)
                            dim = (width, height)

                            # resize image
                            resized = cv2.resize(face, dim, interpolation=cv2.INTER_AREA)
                            cv2.imshow("normalized float64", resized)

                            # create composite image to display each intermediary step
                            composite_img = np.zeros((inp_size[0] * 2, inp_size[1] * 2 * 4, 3), np.uint8)
                            images = [img, stage1, stage2, (stage2.astype(np.uint8) * 256)]
                            # append each face to the composite
                            for i in range(4):
                                composite_img[0:inp_size[0] * 2,
                                i * inp_size[0] * 2:inp_size[0] * 2 + i * inp_size[0] * 2,
                                :] = cv2.resize(images[i], (320, 320), interpolation=cv2.INTER_AREA)
                            cv2.putText(composite_img, "original", (10, 310), cv2.FONT_HERSHEY_PLAIN, 1,
                                        (255, 255, 255), 1)
                            cv2.putText(composite_img, "detected face", (330, 310), cv2.FONT_HERSHEY_PLAIN, 1,
                                        (255, 255, 255), 1)
                            cv2.putText(composite_img, "normalized uint8 direct", (650, 310), cv2.FONT_HERSHEY_PLAIN, 1,
                                        (0, 155, 255), 1)
                            cv2.putText(composite_img, "normalized uint8 256x", (970, 310), cv2.FONT_HERSHEY_PLAIN, 1,
                                        (255, 155, 0), 1)
                            cv2.imshow("classifier preprocessing", composite_img)

                            # hold thingy so cv2 doesn't freak out
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break

                        # run the face through the model and append its results to the predictions for this class set
                        prediction_results.append(main_model.predict(np.expand_dims(face, axis=0))[0])
                    else:
                        # raise an error if something has gone bad
                        raise ValueError("Sample image does not appear to contain face! (", img_path, ")")

            if len(prediction_results) > 0:
                # normalize results to finalized the classifier for this class
                preds = tf.reduce_sum(prediction_results, axis=0)
                # not sure if l2 normalization is needed here
                # (https://www.tensorflow.org/api_docs/python/tf/keras/utils/normalize)
                preds = tf.keras.utils.normalize(np.expand_dims(preds, axis=0), order=2)[0]
                completed_classes[possible_class] = preds

    # save finished classifier in binary format
    pickle.dump(completed_classes, open('output/my_classifier.pkl', 'bw'))

    # clean up cv2 windows
    if verbose:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    exit()
