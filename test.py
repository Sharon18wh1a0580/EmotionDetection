import cv2
from keras.models import model_from_json  
from keras.preprocessing import image 
import numpy as np 

# Load the cascade
#load model  
model = model_from_json(open("fer.json", "r").read())  
#load weights  
model.load_weights('fer.h5')  

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# To capture video from webcam. 
cap = cv2.VideoCapture(0)
# To use a video file as input 
# cap = cv2.VideoCapture('filename.mp4')

while True:
    # Read the frame
    _, img = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.5, 3)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray=gray[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image  
        roi_gray=cv2.resize(roi_gray,(48,48))  
        img_pixels = image.img_to_array(roi_gray)  
        img_pixels = np.expand_dims(img_pixels, axis = 0)  
        img_pixels /= 255  
  
        predictions = model.predict(img_pixels)  
  
        #find max indexed array  
        max_index = np.argmax(predictions[0])  
  
        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise','neutral')  
        predicted_emotion = emotions[max_index]  
  
        cv2.putText(img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)  
  
    resized_img = cv2.resize(img, (1000, 700))  
    cv2.imshow('Facial emotion analysis ',resized_img) 
    # Display
   
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# Release the VideoCapture object
cap.release()