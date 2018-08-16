---
title: Step by step guide 
layout: post
author: anusha.lihala
permalink: /step-by-step-guide/
source-id: 1apSbdw8YEZVn5s4ECYYAPxQ8HDe7WVDGiuRFgReNyS4
published: true
---
## Step by Step Guide to Reproduce Results

* Install OpenCV. In Anaconda environment the following can be used;

```

conda install opencv

```

* Import OpenCV

```

import cv2

```

* Take path of pre-trained Haar frontal face classifier XML file, path of video file (optional), and whether output should be saved as input.

* Read video using VideoCapture class;

```

# if a video path was not supplied, use webcam

if video_path=="":

	camera = cv2.VideoCapture(0)

# otherwise, load the video

else:

	camera = cv2.VideoCapture(video_path)

```

* Initialise VideoWriter class if output is to be saved;

```

fourcc = cv2.VideoWriter_fourcc(*'DIVX')

out = cv2.VideoWriter(fname+'.avi',fourcc, 20.0, (640,480))

```

* Construct classifier

```

face_classifier = cv2.CascadeClassifier(cascade_path)

```

* Continuously read frames from video while they are available;

```

(successful, frame) = camera.read()

if not successful:

break

```

* Convert frame to grayscale;

```

gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

```

* Detect faces in grayscale frame using classifier;

```

face_rects = face_classifier.detectMultiScale(gray_frame, scaleFactor = 1.2, minNeighbors = 5, minSize = (30, 30))

```

* Draw rectangles returned by classifier over frame and display on screen;

```

for (fX, fY, fW, fH) in face_rects:

	cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), (0, 255, 0), 2)

```

* Write frame to output video file if output is to be saved;

```

out.write(frame)

```

* Provide a way to stop the code (here it is stopped by pressing ENTER which has code=13)

```

if cv2.waitKey(1) == 13:

	break

```

* Finally, release all resources

```

camera.release()

if save:

    out.release()

cv2.destroyAllWindows()

```

