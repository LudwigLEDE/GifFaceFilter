import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageSequence

# Load GIF and extract frames
gif_path = '/mnt/data/glitch.gif'
gif = Image.open(gif_path)
frames = [frame.copy() for frame in ImageSequence.Iterator(gif)]

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Capture video from webcam
cap = cv2.VideoCapture(0)

frame_index = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to find face landmarks
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get the coordinates of the nose tip (landmark 1)
            h, w, _ = frame.shape
            nose_tip = face_landmarks.landmark[1]
            nose_x, nose_y = int(nose_tip.x * w), int(nose_tip.y * h)

            # Resize the GIF frame to fit the face (adjust size as needed)
            gif_frame = frames[frame_index]
            gif_frame = gif_frame.resize((100, 100))  # Example size

            # Convert the GIF frame to a format suitable for OpenCV
            gif_frame = cv2.cvtColor(np.array(gif_frame), cv2.COLOR_RGBA2BGRA)

            # Overlay the GIF frame on the webcam frame
            overlay_frame = frame.copy()
            y1, y2 = nose_y - 50, nose_y + 50
            x1, x2 = nose_x - 50, nose_x + 50

            alpha_s = gif_frame[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s

            for c in range(3):
                overlay_frame[y1:y2, x1:x2, c] = (alpha_s * gif_frame[:, :, c] +
                                                  alpha_l * overlay_frame[y1:y2, x1:x2, c])

            frame = overlay_frame

    # Display the result
    cv2.imshow('Face Filter', frame)

    frame_index = (frame_index + 1) % len(frames)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
