import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'images'
images = []
names = []
dirs = os.listdir(path)

for dir in dirs:
    img = cv2.imread(f'{path}/{dir}')
    images.append(img)
    names.append(os.path.splitext(dir)[0])

img = face_recognition.load_image_file('images/musk.jpg')
img2 = face_recognition.load_image_file('images/musk2.jpg')


def findEncodings(images):
    encodeList = []
    for img in images:
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(name):
    with open("Attendace_list.csv", "r+") as f:
        data = f.readlines()
        namelist = []
        for line in data:
            entry = line.split(",")
            namelist.append(entry[0])
        if name not in namelist:
            now = datetime.now()
            dt = now.strftime("%H:%H:%S")
            f.writelines(f"\n{name},{dt}")


encodings = findEncodings(images)

cap = cv2.VideoCapture(0)

while True:
    res, img = cap.read()
    imgs = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    cur_loc = face_recognition.face_locations(imgs)
    cur_encode = face_recognition.face_encodings(imgs, cur_loc)

    for encodedface, faceloc in zip(cur_encode, cur_loc):
        matches = face_recognition.compare_faces(encodings, encodedface)
        print("matches", matches)
        dis = face_recognition.face_distance(encodings, encodedface)
        matchIndex = np.argmin(dis)
        print("matchIndex", matchIndex)
        if matches[matchIndex]:
            name = names[matchIndex].upper()
            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
