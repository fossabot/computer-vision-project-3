"""
File: generate_classifier.py
Created by Andrew Ingson & Chase Carbaugh (aings1@umbc.edu & chasec1@umbc.edu)
Date: 4/19/2021
CMSC 491 Special Topics - Computer Vision

"""

import argparse
import hashlib
import pickle
import time

import cv2
import numpy as np
import progressbar

from config import *


def check_changed(input_dir=input_directory):
    # load old classifier versions
    classifier_hist = dict()
    if os.path.exists(classifier_hist_path):
        classifier_hist = pickle.load(open(classifier_hist_path, 'rb'))
    classifier_hash = dict()
    if os.path.exists(classifier_hash_path):
        classifier_hash = pickle.load(open(classifier_hash_path, 'rb'))

    num_classes, unchanged_classes = 0, 0
    for possible_class in os.listdir(input_dir):
        if "." not in possible_class:
            num_classes += 1
            combined_class_dir = os.path.join(input_dir, possible_class)
            # logging.debug("\n\nTraining on %d for %s", int(len(os.listdir(combined_class_dir))), possible_class)

            # check early exit
            try:
                if classifier_hist.get(possible_class, -1) == len(os.listdir(combined_class_dir)):
                    # create hash of all image names and compare it to the old one
                    files = ""
                    for source_image in os.listdir(combined_class_dir):
                        files += source_image
                    file_hash = hashlib.md5(files.encode()).hexdigest()
                    if file_hash == classifier_hash.get(possible_class, -1):
                        unchanged_classes += 1

            except KeyError:
                time.sleep(0)

    if num_classes == unchanged_classes:
        return False
    else:
        return True


def classify(detailed_output=False):
    # input_directory = 'C:\\Users\\aings\\Downloads\\lfw-funneled\\lfw_funneled'
    # input_directory = 'C:\\Users\\aings\\Downloads\\lfw-a\\lfw'
    # hold references to completed classes
    completed_classes = dict()

    # load old classifier versions
    classifier_hist = dict()
    if os.path.exists(classifier_hist_path):
        classifier_hist = pickle.load(open(classifier_hist_path, 'rb'))
    classifier_hash = dict()
    if os.path.exists(classifier_hash_path):
        classifier_hash = pickle.load(open(classifier_hash_path, 'rb'))
    old_classifier = dict()
    if os.path.exists(classifier_target):
        old_classifier = pickle.load(open(classifier_target, 'rb'))

    progressbar.streams.wrap_stderr()

    if detailed_output:
        print("\n")
    bar = progressbar.ProgressBar(max_value=len(os.listdir(input_directory)), redirect_stderr=True)
    bar.start()
    class_count = 0

    for possible_class in os.listdir(input_directory):
        class_count += 1
        if "." not in possible_class:
            combined_class_dir = os.path.join(input_directory, possible_class)
            # logging.debug("\n\nTraining on %d for %s", int(len(os.listdir(combined_class_dir))), possible_class)

            # check early exit
            try:
                if classifier_hist.get(possible_class, -1) == len(os.listdir(combined_class_dir)):
                    # create hash of all image names and compare it to the old one
                    files = ""
                    for source_image in os.listdir(combined_class_dir):
                        files += source_image
                    file_hash = hashlib.md5(files.encode()).hexdigest()
                    if file_hash == classifier_hash.get(possible_class, -1):
                        if detailed_output:
                            print("Skipping", possible_class, "since it hasn't changed. HASH:", file_hash)
                        completed_classes[possible_class] = old_classifier[possible_class]

                        continue
            except KeyError:
                time.sleep(0)
                # verbose = verbose
            # set up variables for inner loop
            prediction_results = []
            img_count = 0
            files = ""
            # TODO: optimization: save versions of the input images that are just the face? Would speed things up if we
            #  are using the standard input images
            for source_image in os.listdir(combined_class_dir):
                files += source_image
                # skip readme files since they are not images
                if "README" not in source_image:
                    # update stats
                    # print(str(img_count) + ":", source_image)
                    img_count += 1
                    # pull the image from the source directory
                    img_path = os.path.join(combined_class_dir, source_image)
                    img = cv2.imread(img_path)

                    # resize inputs if the source is bigger, THIS STEP MAKES THINGS GO FAST!
                    if resize_inputs and len(img) > resize_target[0] and len(img[0]) > resize_target[1]:
                        img = cv2.resize(img, resize_target, interpolation=cv2.INTER_AREA)

                    # get face location from mtcnn
                    results = detector.detect_faces(img)
                    # if an image is found
                    try:
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
                            if detailed_output:
                                # display float64 image
                                scale_percent = 200  # percent of original size
                                width = int(face.shape[1] * scale_percent / 100)
                                height = int(face.shape[0] * scale_percent / 100)
                                dim = (width, height)

                                # resize image
                                resized = cv2.resize(face, dim, interpolation=cv2.INTER_AREA)
                                cv2.putText(resized, possible_class, (10, 20), cv2.FONT_HERSHEY_PLAIN, 1,
                                            (255, 255, 255), 1)
                                cv2.imshow("normalized float64", resized)

                                # create composite image to display each intermediary step
                                composite_img = np.zeros((inp_size[0] * 2, inp_size[1] * 2 * 4, 3), np.uint8)
                                images = [img, stage1, stage2, (stage2.astype(np.uint8) * 256)]
                                # append each face to the composite
                                for i in range(4):
                                    # CV2 Resize: INTER_CUBIC is better quality, INTER_LINEAR is faster
                                    composite_img[0:inp_size[0] * 2,
                                    i * inp_size[0] * 2:inp_size[0] * 2 + i * inp_size[0] * 2,
                                    :] = cv2.resize(images[i], (320, 320), interpolation=cv2.INTER_LINEAR)
                                cv2.putText(composite_img, "original", (10, 310), cv2.FONT_HERSHEY_PLAIN, 1,
                                            (255, 255, 255), 1)
                                cv2.putText(composite_img, "detected face", (330, 310), cv2.FONT_HERSHEY_PLAIN, 1,
                                            (255, 255, 255), 1)
                                cv2.putText(composite_img, "normalized uint8 direct", (650, 310),
                                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 155, 255), 1)
                                cv2.putText(composite_img, "normalized uint8 256x", (970, 310), cv2.FONT_HERSHEY_PLAIN,
                                            1, (255, 155, 0), 1)
                                cv2.putText(composite_img, possible_class, (10, 20), cv2.FONT_HERSHEY_PLAIN, 1,
                                            (255, 255, 255), 1)
                                cv2.putText(composite_img, source_image, (10, 40), cv2.FONT_HERSHEY_PLAIN, 1,
                                            (255, 255, 255), 1)
                                cv2.putText(composite_img,
                                            str(img_count) + " of " + str(len(os.listdir(combined_class_dir))),
                                            (10, 60), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
                                cv2.imshow("classifier preprocessing", composite_img)

                                # hold thingy so cv2 doesn't freak out
                                if cv2.waitKey(1) & 0xFF == ord('q'):
                                    break

                            # run the face through the model and append its results to the predictions for this class
                            # set
                            prediction_results.append(main_model.predict(np.expand_dims(face, axis=0))[0])
                        else:
                            # raise an error if something has gone bad
                            raise ValueError("Sample image does not appear to contain face! (" + img_path + ")")
                    except ValueError as err:
                        # print(err)
                        continue

            # update classification history
            classifier_hist[possible_class] = len(os.listdir(combined_class_dir))
            classifier_hash[possible_class] = hashlib.md5(files.encode()).hexdigest()
            if len(prediction_results) > 0:
                # normalize results to finalized the classifier for this class
                preds = tf.reduce_sum(prediction_results, axis=0)
                # not sure if l2 normalization is needed here
                # (https://www.tensorflow.org/api_docs/python/tf/keras/utils/normalize)
                preds = tf.keras.utils.normalize(np.expand_dims(preds, axis=0), order=2)[0]
                completed_classes[possible_class] = preds

        bar.update(class_count)
    bar.finish()

    # save finished classifier in binary format
    pickle.dump(completed_classes, open(classifier_target, 'bw'))
    # save updated versions of the old things
    pickle.dump(classifier_hash, open(classifier_hash_path, 'bw'))
    pickle.dump(classifier_hist, open(classifier_hist_path, 'bw'))

    # clean up cv2 windows
    if detailed_output:
        cv2.destroyAllWindows()

    # time.sleep(10)
    # for i in range(100000000):
    #     i = i
    # return completed_classes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classifier Generator')
    parser.add_argument("-v", "--verbosity", type=int, help="increase output verbosity", default=0)
    args = parser.parse_args()

    if args.verbosity > 0:
        verbose = True
    else:
        verbose = False

    classify(verbose)
    exit()
