import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


def preprocess_eye_image(eye_img):
    # Convert eye image to grayscale
    eye_gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)

    # Apply image processing techniques (e.g., thresholding, smoothing) as needed
    _, eye_thresh = cv2.threshold(eye_gray, 30, 255, cv2.THRESH_BINARY)
    eye_smoothed = cv2.GaussianBlur(eye_thresh, (5, 5), 0)

    # Resize the eye image and extract features
    resized_eye_img = cv2.resize(eye_smoothed, (64, 64))
    eye_features = np.reshape(resized_eye_img, (1, -1))

    return eye_features


# Load pre-trained models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

# Initialize classifier for eye tracking
eye_tracking_classifier = SVC()
scaler = StandardScaler()

dataset_dir = 'BioID-FaceDatabase-V1.2'

train_eye_images = []  # To store the loaded eye images

# Load eye images from the dataset directory
for filename in os.listdir(dataset_dir):
    if filename.endswith('.pgm'):
        img_path = os.path.join(dataset_dir, filename)
        eye_img = cv2.imread(img_path)
        train_eye_images.append(eye_img)

# Convert the list of eye images to a numpy array
train_eye_images = np.array(train_eye_images)

# Prepare the eye features for training
train_eye_features = np.concatenate([preprocess_eye_image(img) for img in train_eye_images])

# Fit the scaler with the training data
scaler.fit(train_eye_features)
# Prepare the corresponding labels for training (you need to define the labels based on your dataset)
train_labels = np.array([i for i in range(0, 1521)])

'''
# Train the eye tracking classifier
eye_tracking_classifier.fit(train_eye_features, train_labels)
'''

filepath = 'eyetracking.sav'

eye_tracking_classifier = pickle.load(open(filepath, 'rb'))


# Function to detect faces and eyes
def detect_faces_eyes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    gaze_directions = []  # Store gaze directions for each eye detected
    gaze_detected = False

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Detect eyes within the face region
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        # Perform eye tracking on the eye regions
        for (ex, ey, ew, eh) in eyes:
            eye_img = roi_color[ey:ey + eh, ex:ex + ew]

            # Preprocess the eye image for feature extraction
            eye_features = preprocess_eye_image(eye_img)

            # Normalize features using scaler
            eye_features_normalized = scaler.transform(eye_features)

            # Predict the gaze direction using the eye tracking classifier
            gaze_direction = eye_tracking_classifier.predict(eye_features_normalized)
            gaze_directions.append(gaze_direction.reshape(-1))

            # Display the gaze direction on the frame
            cv2.putText(frame, "Gaze: {}".format(gaze_direction), (x + ex, y + ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

            if len(gaze_direction) > 0:  # Check if gaze direction is not empty
                gaze_detected = True

    face_detected = len(faces) > 0
    last_gaze_direction = gaze_directions[-1] if gaze_detected else None

    return frame, face_detected, last_gaze_direction


def run():
    # Initialize variables
    telemetry = []
    telemetry_time = []
    unit_time = 5  # Time interval for telemetry collection (in seconds)

    # Open webcam
    cap = cv2.VideoCapture(0)
    total_face = 0
    total_gaze = 0
    face_start_time = None
    gaze_start_time = None
    face_duration = 0
    gaze_duration = 0

    init_time = time.time()

    while True:
        # Read frame from webcam
        ret, frame = cap.read()

        # Break the loop if no frame is captured
        if not ret:
            break

        # Detect faces and gaze in the frame
        output_frame, face_detected, gaze_direction1 = detect_faces_eyes(frame)

        # Update face duration
        if face_detected:
            if face_start_time is None:
                face_start_time = time.time()
        else:
            if face_start_time is not None:
                face_duration = time.time() - face_start_time
                total_face += face_duration
                face_start_time = None

        if gaze_direction1 is not None:
            if gaze_start_time is None:
                gaze_start_time = time.time()
        else:
            if gaze_start_time is not None:
                gaze_duration = time.time() - gaze_start_time
                total_gaze += gaze_duration
                gaze_start_time = None

        # Display the output frame
        cv2.imshow('Face and Eye Detection', output_frame)

        # Log the duration of face and gaze
        #print("Face duration: {:.2f}s".format(face_duration))
        #print("Gaze duration: {:.2f}s".format(gaze_duration))

        current_time = time.time()
        elapsed_time = current_time - init_time

        if elapsed_time >= unit_time:
            telemetry.append([face_duration, gaze_duration])
            telemetry_time.append(current_time)
            init_time = current_time

        # Break the loop if 's' is pressed
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

    current_time = time.time()
    if face_start_time is not None:
        total_face += current_time - face_start_time

    print("\n\nTotal Face duration: {:.2f}s".format(total_face))
    print("Total Gaze duration: {:.2f}s".format(total_gaze))

    # Plot telemetry
    telemetry = np.array(telemetry)
    telemetry_time = np.array(telemetry_time)
    plt.plot(telemetry_time, telemetry[:, 0], label='Face Duration')
    plt.plot(telemetry_time, telemetry[:, 1], label='Gaze Duration')
    plt.xlabel('Time')
    plt.ylabel('Duration')
    plt.legend()
    #plt.show()

    # Save telemetry to a CSV file
    telemetry_file = 'telemetry.csv'
    np.savetxt(telemetry_file, np.column_stack((telemetry_time, telemetry)), delimiter=',')

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()
