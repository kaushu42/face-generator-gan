import os
import cv2

# Load the haar cascade to detect the faces
face_classifier = cv2.CascadeClassifier(
    'cascades/haarcascade_frontalface_default.xml')


def detect_faces(image):
    '''
            Detect the faces in the input image.
            The image is converted to a grayscale image, then the cascade is applied.

            Input: An input image
            Returns: A generator that yields all the faces in the input image
    '''
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(grey, 1.5, 5)
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        yield face


# Read the video files from the videos directory
VIDEO_DIR = 'videos'
input_files = os.listdir(VIDEO_DIR)
input_files.remove('help.txt')

IMAGE_DIR = 'inputs'

if not len(input_files):
    print('No videos in video directory. Please add some files')
    exit()
for file_count, input_file in enumerate(input_files):
    cap = cv2.VideoCapture(os.path.join(VIDEO_DIR, input_file))
    frame_count = 0
    print(input_file)
    while True:
        _, frame = cap.read()
        if frame_count % 5 == 0:
            images = detect_faces(frame)
            for i, image in enumerate(images):
                resized = cv2.resize(image, (81, 81))
                file = os.path.join(
                    IMAGE_DIR, f'{file_count}-{frame_count}-{i}.png')
                cv2.imwrite(file, resized)
        frame_count += 1
        if frame is None:
            print('END')
            break
    cap.release()
    cv2.destroyAllWindows()
