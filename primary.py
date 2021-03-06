"""
File: primary.py
Created by Andrew Ingson & Chase Carbaugh (aings1@umbc.edu & chasec1@umbc.edu)
Date: 4/15/2021
CMSC 491 Special Topics - Computer Vision

"""

import argparse
import pickle
import random
import string
import time
from multiprocessing import Process

import cv2
import numpy as np
from tensorflow.keras import losses
from tensorflow.keras import utils

import generate_classifier
import rename_class
from config import *


def main():
    print("----- Project 3 -----")

    # TODO: Separate the verbose arguments into more specific parts (e.g. an fps arg)
    parser = argparse.ArgumentParser(description='Adaptive Security Camera')
    parser.add_argument("-v", "--verbosity", type=int, help="increase output verbosity", default=0)
    args = parser.parse_args()

    if args.verbosity > 0:
        verbose = True
    else:
        verbose = False

    # load classifier dictionary back in
    generate_classifier.classify(verbose)
    # load webcam video
    cap = cv2.VideoCapture(0)
    # use video file instead
    # cap = cv2.VideoCapture("C:\\Users\\aings\\Downloads\\C0002.MP4")
    total_frames = 0
    strt = time.time()
    curr_time = time.time()

    # create multithreading stuff
    class_gen_proc = Process(target=generate_classifier.classify(verbose))
    running = False

    unknown_count = 0
    unknown_dir = "new_" + ''.join(random.choice(string.ascii_letters) for i in range(10))

    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        # for fps count
        total_frames += 1

        if running and not class_gen_proc.is_alive():
            class_gen_proc.join()
            unknown_count = 0
            unknown_dir = "new_" + ''.join(random.choice(string.ascii_letters) for i in range(10))
            running = False
            # classifier = class_gen_proc.va
            curr_time = time.time()
            print("Classifier Updated")
            class_gen_proc = Process(target=generate_classifier.classify(verbose))

        classifier = pickle.load(open(classifier_target, 'rb'))
        # get faces from mtcnn
        if resize_frame:
            frame = cv2.resize(frame, (int(cap.get(3)) // 2, int(cap.get(4)) // 2), interpolation=cv2.INTER_AREA)
        results = detector.detect_faces(frame)
        for data in results:
            # update curr_time since we've found a face
            curr_time = time.time()
            # I'm suspecting this value might need to be higher than we think
            if data['confidence'] >= 0.994:
                # print("MTCNN Confidence:", data['confidence'])
                # create sub image that contains only the face
                bounding_box = data['box']
                bounding_box[0], bounding_box[1] = abs(bounding_box[0]), abs(bounding_box[1])
                face = frame[bounding_box[1]:bounding_box[1] + bounding_box[3],
                       bounding_box[0]:bounding_box[0] + bounding_box[2]]

                save_face = np.copy(face)
                # get coordinates for the bounding box from mtcnn
                face_topLeft = bounding_box[0], bounding_box[1]
                face_bottomRight = bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]

                # normalize the face image from uint8 to float64 and convert it to the size to give to the model
                face = cv2.resize(face, inp_size)
                face = (face - face.mean()) / face.std()
                # ask our model to predict the face
                embedded_128 = main_model.predict(np.expand_dims(face, axis=0))[0]
                # clean the values up so they work for loss
                embedded_128 = utils.normalize(embedded_128.reshape(1, -1), order=2)[0]

                # vars to represent the best fit face
                loss = 1.0
                found_class = False
                best_clss = ""
                # iterate through all the classes in the classifier to see if the found faces matches them
                for clss in classifier:
                    # using cosine bc we are trying to find similarity between the found face and the classifiers
                    # loss of -1 means absolute similarity, loss of 1 means absolute dissimilarity
                    cos_sim = losses.CosineSimilarity()

                    similarity = cos_sim(classifier[clss], embedded_128).numpy()
                    # if loss is below this value, we found a known face
                    # -.35 seems to work better with masks
                    recognition_thresh = -0.55
                    if similarity < recognition_thresh and similarity < loss:
                        # update tracking vars to current best
                        found_class = True
                        best_clss = clss
                        loss = similarity

                # create labels for the face to go into the frame
                if verbose:
                    # create circles showing the key points
                    keypoints = data['keypoints']
                    radius = 5
                    thickness = -1  # int(radius / 2)
                    cv2.circle(frame, (keypoints['left_eye']), radius, (0, 155, 255), thickness)
                    cv2.circle(frame, (keypoints['right_eye']), radius, (0, 155, 255), thickness)
                    cv2.circle(frame, (keypoints['nose']), radius, (0, 155, 255), thickness)
                    cv2.circle(frame, (keypoints['mouth_left']), radius, (0, 155, 255), thickness)
                    cv2.circle(frame, (keypoints['mouth_right']), radius, (0, 155, 255), thickness)

                # place the bounding box onto the frame
                rec_thicc = 2
                if not found_class:
                    cv2.rectangle(frame, face_topLeft, face_bottomRight, (0, 0, 255), rec_thicc)
                    cv2.rectangle(frame, (face_topLeft[0] - rec_thicc, face_topLeft[1]),
                                  (face_bottomRight[0] + rec_thicc, face_topLeft[1] - 25), (0, 0, 255),
                                  -1)
                    cv2.putText(frame, "unrecognized", face_topLeft, cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
                    if verbose:
                        cv2.putText(frame, f' Loss: {loss:.2f}', (face_topLeft[0], face_bottomRight[1] + 20),
                                    cv2.FONT_HERSHEY_PLAIN, 1.5,
                                    (255, 255, 255), 2)
                        cv2.putText(frame, f' Conf: {data["confidence"]:.7f}',
                                    (face_topLeft[0], face_bottomRight[1] + 40),
                                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

                    # save image of unknown face
                    new_img_name = ''.join(random.choice(string.ascii_letters) for i in range(10)) + ".jpg"
                    img_path = 'input/' + unknown_dir
                    if not os.path.exists(img_path):
                        os.mkdir(img_path)
                    # check to see if unknown is full
                    files = os.listdir(img_path)
                    # save face to current unknown directory
                    if len(files) < face_limit and unknown_count % face_rate == 0:
                        unknown_count = 0
                        cv2.imwrite(img_path + '/' + new_img_name, save_face)
                        if verbose:
                            cv2.putText(frame, "Saved Face", (10, 100),
                                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                    unknown_count += 1

                else:
                    cv2.rectangle(frame, face_topLeft, face_bottomRight, (0, 255, 0), rec_thicc)
                    cv2.rectangle(frame, (face_topLeft[0] - rec_thicc, face_topLeft[1]),
                                  (face_bottomRight[0] + rec_thicc, face_topLeft[1] - 25), (0, 255, 0),
                                  -1)
                    cv2.putText(frame, best_clss, (face_topLeft[0], face_topLeft[1] - 5), cv2.FONT_HERSHEY_PLAIN, 1.5,
                                (0, 0, 0), 2)
                    if verbose:
                        cv2.putText(frame, f' Loss: {loss:.2f}', (face_topLeft[0], face_bottomRight[1] + 20),
                                    cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2)
                        cv2.putText(frame, f' Conf: {data["confidence"]:.7f}',
                                    (face_topLeft[0], face_bottomRight[1] + 40),
                                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

                    new_img_name = ''.join(random.choice(string.ascii_letters) for i in range(10)) + ".jpg"
                    img_path = 'input/' + best_clss
                    # check to see if target is full
                    files = os.listdir(img_path)
                    if len(files) < face_limit:
                        cv2.imwrite(img_path + '/' + new_img_name, save_face)

        # update classifier
        if time.time() - curr_time > inactivity_thresh and not running and generate_classifier.check_changed():
            if verbose:
                print("Updating Classifier")
            class_gen_proc.start()
            running = True
            # classifier =
            # generate_classifier.classify(verbose)
            # curr_time = time.time()
        if verbose:
            # show fps
            fps = total_frames / (time.time() - strt)
            best_clr = ~int(frame[10, 20, 0]), ~int(frame[10, 20, 1]), ~int(frame[10, 20, 2])
            cv2.putText(frame, f'Source FPS: {cap.get(cv2.CAP_PROP_FPS):.2f}', (10, 20), cv2.FONT_HERSHEY_PLAIN, 1,
                        best_clr, 1)
            cv2.putText(frame, f'Processed FPS: {fps:.2f}', (10, 40), cv2.FONT_HERSHEY_PLAIN, 1, best_clr, 1)
            cv2.putText(frame, "Source: " + str(cap.get(3)) + "x" + str(cap.get(4)), (10, 60),
                        cv2.FONT_HERSHEY_PLAIN, 1, best_clr, 1)
            if running:
                cv2.putText(frame, "Updating Classifier", (10, 80),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        # quit if q key is pressed
        if cv2.waitKey(1) == ord('q'):
            break
        # rename classes if a key is pressed
        if cv2.waitKey(33) == ord('a'):
            rename_class.rename_class_dir(ignore_dir=unknown_dir)
            generate_classifier.classify(verbose)
            curr_time = time.time()

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    exit()
