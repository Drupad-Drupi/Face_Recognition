# Face Recognition System

This project is a face recognition system using OpenCV and Python. It consists of multiple scripts for gathering data, training a model, and recognizing faces in real-time.

## Prerequisites

Ensure you have the following dependencies installed:

- Python 3.x
- OpenCV (`cv2`)
- NumPy
- PIL (Pillow)

You can install the dependencies using:
```bash
pip install opencv-python numpy pillow
```

## Project Files

1. **camera_test.py** - Tests the camera to ensure it works properly.
2. **Data_gathering.py** - Captures face images from the webcam and stores them for training.
3. **face_detection.py** - Detects faces in real-time using Haar cascades.
4. **face_reconize.py** - Recognizes trained faces using the model.
5. **face_trainer.py** - Trains the face recognition model based on collected data.
6. **haarcascade_frontalface_default.xml** - Pre-trained XML file for detecting faces.

## Usage

### 1. Test Camera
Run the following command to check if the camera is working:
```bash
python camera_test.py
```

### 2. Gather Face Data
To collect face images for training, run:
```bash
python Data_gathering.py
```
Follow the instructions on the screen to capture images.

### 3. Train the Model
Once data is collected, train the model using:
```bash
python face_trainer.py
```

### 4. Recognize Faces
To recognize faces in real-time, run:
```bash
python face_reconize.py
```
This will open a camera window and attempt to recognize stored faces.

## Notes
- Ensure good lighting conditions for better recognition.
- The more images collected, the better the accuracy.
- You can modify `haarcascade_frontalface_default.xml` with a different Haar cascade if needed.

## License
This project is open-source and free to use.

## Author
Drupad G

