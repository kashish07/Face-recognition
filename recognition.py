import numpy as np
import cv2
url = "http://192.168.43.1:8080/video"
cam = cv2.VideoCapture(0)
face_cas = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
font = cv2.FONT_ITALIC
f1 = np.load('Name.npy').reshape((30,50*50*3))
f2 = np.load('Name2.npy').reshape((30,50*50*3))

names = {
    0 : 'Name',
    1 : 'Name2'  ,  
        
        }
label = np.zeros((100,1))
label[50:100 ,:]= 1
 
data = np.concatenate([f1 , f2   ])
def distance (x1,x2):
    return np.sqrt(((x1-x2)**2).sum())
def knn (x,train,labels,k=5 ):
    m= train.shape[0]
    dist = []
    for ix in range(m):
        dist.append(distance(x,train[ix]))
    dist = np.array(dist)
    indx = np.argsort(dist)
    sorted_labels = labels[indx][:k]
    counts = np.unique(sorted_labels,return_counts =True)
    return counts[0][np.argmax(counts[1])] 
#from sklearn.neighbors import KNeighborsClassifier
#l = KNeighborsClassifier(neighbors =5)
#l.fit( )
while True:
    ret , frame = cam.read()
    if ret==True:
        gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
        faces = face_cas.detectMultiScale(gray , 1.3 , 5)
        
        for (x,y,w,h) in faces:
            face_component = frame[y:y+h , x:x+w,:]
            fc = cv2.resize(face_component, (50,50))
            lab = knn(fc.flatten(),data,label)
            text = names[int(lab)]
            cv2.putText(frame , text , (x,y),font,1 ,(255,0,2),2)
            cv2.rectangle(frame , (x,y),(x+w,y+h),(0,225,0),2)
        cv2.imshow('frame' , frame)    
        if cv2.waitKey(1)==27:
            break
    else:
        print("error")
cv2.destroyAllWindows()
    