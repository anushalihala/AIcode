---
title: Methodology
layout: post
author: anusha.lihala
permalink: /methodology/
source-id: 13kPQrUhsz2lLeqtKvlHANDBuwBeWHs05tLeL4-1g7-M
published: true
---
# Methodology

The pre-trained Haar frontal face classifier provided by OpenCV was used to detect faces.

The Haar classifier was chosen instead of the LBP classifier due to its greater accuracy.

Here, Haar features are extracted with the help of a convolution window and Adaboost is used to select the features with minimum error rate.

**Process followed by code:**

* OpenCV's VideoCapture class is used to read video from file or webcam.

* Frames in video are read one at a time

* Each frame is;

    * Converted to grayscale (as the classifiers process grayscale images)

    * Grayscale frame is passed to classifier's detectMultiScale method, and parameters tuned;

        * scaleFactor; Scaling used to resize input image at each step, decreased to increase the chance of a matching size with the model for detection / increased to reduce false positives.

        * minNeighbors; Number of rectangles required in neighbourhood for region to be classified as face, increased to reduce false positives.

        * minSize; Set to be smaller than size of face in image

    * Rectangular coordinates returned by detectMultiScale method drawn on top of frame and shown on screen.

## Limitations

The algorithm can only detect frontal faces, due to the pre-trained classifier used.

## Future Improvements

Deep Learning model with OpenCV, reference article;

[https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/](https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/)

Benefits: Higher accuracy, Greater flexibility - can detect faces from more angles

