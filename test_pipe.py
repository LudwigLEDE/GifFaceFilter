import mediapipe as mp
import cv2

# Open a video capture object
cap = cv2.VideoCapture(0)

# Create a mediapipe hand detection object
hands = mp.solutions.hands.Hands()

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hands
    results = hands.process(frame_rgb)

    # Print the hand landmarks if any are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            i = 0
            for landmark in hand_landmarks.landmark:
                # Display the landmark
                cv2.circle(frame, (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])), 5, (0, 255, 0), -1)
                if i == 0:
                    print(i,landmark.z)
                i += 1

    # Display the frame
    cv2.imshow('Frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()

# Release the hands object
hands.close()

# Destroy all windows
cv2.destroyAllWindows()