# Real time Face recognition

modules you need to run the scripts 
  - numpy ( to array calculations and manipulations)  
   - opencv ( for image processing and use of haarcascade (for face detection )


Face recognition via k nearest neighbour algorithm \
There are two scripts in the project -
 - face_data.py
  - recognition.py
  
## face_data.py   
run this script to generate data of face , edit the name in the script to your desired display name , also you can edit the no of images you want to capture . \
the data will be stored in the form of numpy array in the same location as your script

## recognition.py
This script actually performs the recognition . Change the name in the dictionary to same as your data file name. \
also you can choose your source for the video feed , in video capture 0 is for default webcam , modify url to your video link and in videocapture pass 1 as argument.
