import cv2

cascade_path = input("Path to where the face cascade file resides: ")
video_path = input("Path to the (optional) video file [Leave blank if webcam to be used]: ")
while(True):
	save = input("Save output? [y/n]: ")
	if save=="y":
		save=True
		fname = input("Enter name of output file: ")
		break
	elif save=="n":
		save=False
		break
print("Starting face recognition.. (Please press ENTER to exit)")

		
# if a video path was not supplied, use webcam
if video_path=="":
	camera = cv2.VideoCapture(0)
# otherwise, load the video
else:
	camera = cv2.VideoCapture(video_path)
    
if save:
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(fname+'.avi',fourcc, 20.0, (640,480))
    
#initialise classifier 
face_classifier = cv2.CascadeClassifier(cascade_path)


while True:
	# grab frame
	(successful, frame) = camera.read()

	if not successful:
		break

	#convert to grayscale
	gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the image 
	face_rects = face_classifier.detectMultiScale(gray_frame, scaleFactor = 1.2, minNeighbors = 5,
		minSize = (30, 30))

	# loop over the face bounding boxes and draw them
	for (fX, fY, fW, fH) in face_rects:
		cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), (0, 255, 0), 2)

	# show frame with faces marked
	cv2.imshow("Face", frame)
    
	if save:
		#write frame to output video
		out.write(frame)

	# exit loop if 'enter' key is pressed
	if cv2.waitKey(1) == 13:
		break

# release resources
camera.release()
if save:
    out.release()
cv2.destroyAllWindows()