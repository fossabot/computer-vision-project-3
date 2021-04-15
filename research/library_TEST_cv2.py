"""
File: library_TEST_cv2.py
Created by Andrew Ingson (aings1@umbc.edu)
Date: 4/15/2021
CMSC 491 Special Topics - Computer Vision

https://github.com/linxiaohui/mtcnn-opencv

"""

import cv2
from mtcnn_cv2 import MTCNN

detector = MTCNN()
test_pic = "_DSC9940-Edit.jpg"

image = cv2.cvtColor(cv2.imread(test_pic), cv2.COLOR_BGR2RGB)
result = detector.detect_faces(image)

# Result is an array with all the bounding boxes detected. Show the first.
print(result)

if len(result) > 0:
    keypoints = result[0]['keypoints']
    bounding_box = result[0]['box']

    cv2.rectangle(image,
                  (bounding_box[0], bounding_box[1]),
                  (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                  (0, 155, 255),
                  20)
    radius = 50
    thickness = int(radius/2)
    cv2.circle(image, (keypoints['left_eye']), radius, (0, 155, 255), thickness)
    cv2.circle(image, (keypoints['right_eye']), radius, (0, 155, 255), thickness)
    cv2.circle(image, (keypoints['nose']), radius, (0, 155, 255), thickness)
    cv2.circle(image, (keypoints['mouth_left']), radius, (0, 155, 255), thickness)
    cv2.circle(image, (keypoints['mouth_right']), radius, (0, 155, 255), thickness)

    scale_percent = 40  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    cv2.imwrite("result.jpg", cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))

with open(test_pic, "rb") as fp:
    marked_data = detector.mark_faces(fp.read())
with open("marked.jpg", "wb") as fp:
    fp.write(marked_data)