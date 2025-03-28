import cv2
import numpy as np
import os
import pyttsx3
import time

# Load the trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

# Load the Haar Cascade for face detection
cascadePath = r"D:\Codes\Project_Face\Face_detection\haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Font for displaying text
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize the ID counter
id = 0

# Names related to IDs
names = ['None', '(facenames)']

# Start video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # Set video width
cam.set(4, 480)  # Set video height

# Define minimum window size for face recognition
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

# Variables to track the last greeting time
last_greet_time = time.time()
last_unknown_time = time.time()
greet_interval = 10  # Time interval for greeting in seconds
last_id = None  # To keep track of the last recognized ID

while True:
    ret, img = cam.read()
    img = cv2.flip(img, 1)  # Flip vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        # Check if confidence is less than 100 (0 is a perfect match)
        if confidence < 100:
            user_name = names[id]
            confidence = "  {0}%".format(round(100 - confidence))

            # Check if it's time to greet the user
            current_time = time.time()
            if last_id != id or (current_time - last_greet_time) >= greet_interval:
                greeting = f"Hello, {user_name}.  how can i help you"
                engine.say(greeting)
                engine.runAndWait()
                last_greet_time = current_time  # Update the last greet time
                last_id = id  # Update the last recognized ID
        else:
            user_name = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

            # Respond to unknown face every 10 seconds
            current_time = time.time()
            if (current_time - last_unknown_time) >= greet_interval:
                unknown_response = "I didn't recognize your face, sorry."
                engine.say(unknown_response)
                engine.runAndWait()
                last_unknown_time = current_time  # Update the last unknown response time

        # Display the name and confidence on the image
        cv2.putText(img, str(user_name), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    # Show the video feed with recognized faces
    cv2.imshow('camera', img)

    # Exit on pressing 'ESC'
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break

# Cleanup
print("\n[INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
