import numpy as np
import cv2
url = "http://192.168.43.1:8080/video"
cam = cv2.VideoCapture(0)
face_cas = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
data = []
ix = 0
n_images = 30
while True:
    ret , frame = cam.read()
    if ret==True:
        gray =cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
        faces = face_cas.detectMultiScale(gray , 1.3 , 5)
        for (x,y,w,h) in faces:
            face_component = frame[y:y+h , x:x+w,:]
            fc = cv2.resize(face_component, (50,50))
            if ix%10 ==0 and len(data)<n_images:
                data.append(fc)
            cv2.rectangle(frame , (x,y) ,(x+w , y+h), (255,0,255),2)
        ix +=1
        cv2.imshow('frame' , frame)
        if cv2.waitKey(1)==27 or len(data)>=n_images:
            break
    else:
        print("error")
cv2.destroyAllWindows()
data = np.array(data)
np.save('Name2',data)