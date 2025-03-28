import cv2
import os
import pyttsx3

# Create a directory for the dataset if it doesn't exist
dataset_path = "dataset"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Initialize video capture
cam = cv2.VideoCapture(0)
cam.set(3, 1280)  # Set video width to 1280
cam.set(4, 720)   # Set video height to 720

# Load the Haar Cascade for face detection
face_detector = cv2.CascadeClassifier(r'Path of file.xml(same one)')

# For each person, enter one numeric face id
face_id = input('\nEnter user ID and press <return> ==>  ')

print("\n[INFO] Initializing face capture. Look at the camera and wait...")

# Initialize individual sampling face count
count = 0

# Allow some time for the camera to adjust
cv2.waitKey(2000)

while True:
    ret, img = cam.read()
    img = cv2.flip(img, 1)  # Flip video image vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30 , 30))

    if len(faces) == 0:
        cv2.putText(img, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1

            # Save the captured image into the datasets folder
            cv2.imwrite(os.path.join(dataset_path, f"User .{face_id}.{count}.png"), gray[y:y + h, x:x + w])

            # Provide feedback on the number of images captured
            cv2.putText(img, f"Images Captured: {count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the image with rectangles and text
    cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
    if k == 27:  # Exit on 'ESC'
        break
    elif count % 10 == 0 and count > 0:  # Prompt for a new expression every 10 images
        engine = pyttsx3.init()
        engine.say("Please change your expression.")
        engine.runAndWait()
        print("Please change your expression.")

    if count >= 50:  # Stop after capturing 50 images
        break

# Cleanup
print("\n[INFO] Scanning completed. Exiting Program and cleanup stuff.")

cam.release()
cv2.destroyAllWindows()
