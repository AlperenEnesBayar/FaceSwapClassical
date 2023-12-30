import cv2
import dlib
import numpy as np

def detect_faces(image, face_cascade):
    # Convert the image to grayscale for face detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)


def extract_facial_landmarks(image, face_detector, landmark_predictor):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_detector(gray_image)

    for face in faces:
        # Predict facial landmarks for each detected face
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)


        landmarks = landmark_predictor(gray_image, face)

        # Draw circles around the facial landmarks
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(image, (x, y), 2, (0, 0, 255), -1)

def replace_face(image, replacement_face, face_detector, landmark_predictor):
    original_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the original image
    faces = face_detector(original_gray)
    faces_rep = face_detector(replacement_face)

    if len(faces) == 0:
        print("No faces found in the original image.")
        return

    # Assume only one face in the original image for simplicity
    original_landmarks = landmark_predictor(original_gray, faces[0])
    for n in range(0, 68):
            x = original_landmarks.part(n).x
            y = original_landmarks.part(n).y
            cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
    
    # Extract coordinates of facial landmarks
    original_landmarks = np.array([(p.x, p.y) for p in original_landmarks.parts()])

    predicted_landmarks = landmark_predictor(cv2.cvtColor(replacement_face, cv2.COLOR_BGR2GRAY), faces_rep[0])
    for n in range(0, 68):
        x = predicted_landmarks.part(n).x
        y = predicted_landmarks.part(n).y
        cv2.circle(replacement_face, (x, y), 2, (0, 0, 255), -1)

    cv2.imshow("replacement_face", replacement_face)
    
    # Assume replacement face image has the same number of facial landmarks
    replacement_landmarks = np.array([(p.x, p.y) for p in predicted_landmarks.parts()])

    # Calculate the affine transformation matrix
    transformation_matrix = cv2.estimateAffinePartial2D(replacement_landmarks, original_landmarks)[0]

    # Apply the affine transformation to the replacement face
    # Extracting the coordinates
    x, y, w, h = faces_rep[0].left(), faces_rep[0].top(), faces_rep[0].width(), faces_rep[0].height()
    aligned_face = cv2.warpAffine(replacement_face[y:y+h, x:x+w], transformation_matrix, (image.shape[1], image.shape[0]))
    cv2.imshow("aligned_face", aligned_face)

    # Blend the aligned face with the original image
    mask = np.where(aligned_face != 0, 255, 0).astype('uint8')
    cv2.imshow("mask", mask)
    image = cv2.seamlessClone(aligned_face, image, mask, (image.shape[1] // 2, image.shape[0] // 2), cv2.NORMAL_CLONE)
    cv2.imshow("seamless", image)




# Replace 'your_image_path.jpg' with the actual path to your image
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
replacement_face = cv2.imread("henry_cavill.jpg")

cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    # Check if the frame was read successfully
    if not ret:
        print("Error: Could not read frame.")
        break
    
    # detect_faces(frame, face_cascade)
    # extract_facial_landmarks(frame, face_detector, landmark_predictor)
    replace_face(frame.copy(), replacement_face.copy(), face_detector, landmark_predictor)


    # Display the frame
    cv2.imshow("Camera", frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
