import os
import cv2

eye_classifier = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')
face_classifier = cv2.CascadeClassifier(
    'cascades/haarcascade_frontalface_default.xml')


def detect_face_and_eyes(image):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(grey, 1.5, 5)
    roi_original = None
    for (x, y, w, h) in faces:
        roi_grey = grey[y:y+h, x:x+w]
        roi_original = image[y:y+h, x:x+w]
        yield roi_original

input_files = os.listdir('videos')
input_files.remove('help.txt')
for input_file in input_files:
	cap = cv2.VideoCapture(input_file)
	count = 0
	filecount = 0
	while True:
	    _, frame = cap.read()
	    count += 1
	    if count % 2 == 0:
	        images = detect_face_and_eyes(frame)
	        for image in images:
	            resized = cv2.resize(image, (81, 81))
	            cv2.imshow('Video', resized)
	            # file = f'images/image-{filecount}.png'
	            # cv2.imwrite(file, resized)
	            # filecount += 1


	cap.release()
	cv2.destroyAllWindows()
