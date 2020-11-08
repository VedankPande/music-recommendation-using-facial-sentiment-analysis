import face_recognition
import cv2
import os
import matplotlib.pyplot as plt

KNOWN_FACES = []
KNOWN_NAMES = []    
UNKNOWN_FACES = []
TOLERANCE = 0.6

KNOWN_PATH = '/home/vedank/Desktop/data/face_data/known_faces'
UNKNOWN_PATH = '/home/vedank/Desktop/data/face_data/unknown_faces'

for name in os.listdir(KNOWN_PATH):
    for image_name in os.listdir(f"{KNOWN_PATH}/{name}"):
        
        image = face_recognition.load_image_file(f"{KNOWN_PATH}/{name}/{image_name}")
        known_face_location = face_recognition.face_locations(image,model="hog")
        face_encodings = face_recognition.face_encodings(image,known_face_location)
        KNOWN_FACES.append(face_encodings[0])
        KNOWN_NAMES.append(name)

for image in os.listdir(UNKNOWN_PATH):

    image = face_recognition.load_image_file(f"{UNKNOWN_PATH}/{image}")

    unknown_face_location = face_recognition.face_locations(image,model="hog")
    unknown_encoding = face_recognition.face_encodings(image,unknown_face_location)
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

    for encoding,location in zip(unknown_encoding,unknown_face_location):

        res = face_recognition.compare_faces(encoding,KNOWN_FACES,TOLERANCE)

        if True in res:
            match = KNOWN_NAMES[(res.index(True))]

            for loc in location:
                top_left = (location[3],location[0])
                bottom_right = (location[1],location[2])
                cv2.rectangle(image,top_left,bottom_right, color=(0,255,0))

                    
            cv2.imshow(f"{match}",image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()