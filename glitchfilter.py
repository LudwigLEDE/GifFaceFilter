import mediapipe as mp
import cv2
import math
import numpy as np
import csv
from PIL import Image

VISUALIZE_FACE_POINTS = False

# Define the glitch filter configuration
filters_config = {
    'glitch': [{'path': "mnt/data/glitch.gif",  # Update this path
                'anno_path': "mnt/data/glitch_annotations.csv",  # Update this path
                'morph': False, 'animated': True, 'has_alpha': True}],
}

# Detect facial landmarks in the image
def getLandmarks(img):
    mp_face_mesh = mp.solutions.face_mesh
    selected_keypoint_indices = [127, 93, 58, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 288, 323, 356, 70, 63, 105, 66, 55,
                                 285, 296, 334, 293, 300, 168, 6, 195, 4, 64, 60, 94, 290, 439, 33, 160, 158, 173, 153, 144, 398, 385,
                                 387, 466, 373, 380, 61, 40, 39, 0, 269, 270, 291, 321, 405, 17, 181, 91, 78, 81, 13, 311, 306, 402, 14,
                                 178, 162, 54, 67, 10, 297, 284, 389]

    height, width = img.shape[:-1]

    with mp_face_mesh.FaceMesh(max_num_faces=1, static_image_mode=True, min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:
            print('Face not detected!!!')
            return None

        for face_landmarks in results.multi_face_landmarks:
            values = np.array(face_landmarks.landmark)
            face_keypnts = np.zeros((len(values), 2))

            for idx, value in enumerate(values):
                face_keypnts[idx][0] = value.x
                face_keypnts[idx][1] = value.y

            # Convert normalized points to image coordinates
            face_keypnts = face_keypnts * (width, height)
            face_keypnts = face_keypnts.astype('int')

            relevant_keypnts = []

            for i in selected_keypoint_indices:
                relevant_keypnts.append(face_keypnts[i])
            return relevant_keypnts

# Helper function to apply filter with alpha channel
def apply_filter_with_alpha(frame, points, filter_runtime):
    filter_img = filter_runtime['image']
    if filter_runtime['animated']:
        filter_img = filter_img[filter_runtime['frame_index']]
        filter_runtime['frame_index'] = (filter_runtime['frame_index'] + 1) % len(filter_runtime['image'])
    
    if filter_img.mode == 'RGBA':
        filter_img = np.array(filter_img)
        
        # Get bounding box around landmarks
        min_x = min(points, key=lambda p: p[0])[0]
        min_y = min(points, key=lambda p: p[1])[1]
        max_x = max(points, key=lambda p: p[0])[0]
        max_y = max(points, key=lambda p: p[1])[1]

        # Resize the filter image to the bounding box size
        h, w = max_y - min_y, max_x - min_x
        filter_img = cv2.resize(filter_img, (w, h), interpolation=cv2.INTER_AREA)
        
        alpha_s = filter_img[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(0, 3):
            frame[min_y:max_y, min_x:max_x, c] = (alpha_s * filter_img[:, :, c] + alpha_l * frame[min_y:max_y, min_x:max_x, c])
    
    return frame


# Helper function to apply filter with alpha channel
def apply_filter_with_alpha(frame, points, filter_runtime):
    filter_img = filter_runtime['image']
    if filter_runtime['animated']:
        filter_img = filter_img[filter_runtime['frame_index']]
        filter_runtime['frame_index'] = (filter_runtime['frame_index'] + 1) % len(filter_runtime['image'])
    
    if filter_img.mode == 'RGBA':
        filter_img = np.array(filter_img)
        
        # Get bounding box around landmarks
        min_x = min(points, key=lambda p: p[0])[0]
        min_y = min(points, key=lambda p: p[1])[1]
        max_x = max(points, key=lambda p: p[0])[0]
        max_y = max(points, key=lambda p: p[1])[1]

        # Ensure the bounding box is within frame dimensions
        min_x = max(min_x, 0)
        min_y = max(min_y, 0)
        max_x = min(max_x, frame.shape[1])
        max_y = min(max_y, frame.shape[0])

        # Resize the filter image to the bounding box size
        h, w = max_y - min_y, max_x - min_x
        filter_img = cv2.resize(filter_img, (w, h), interpolation=cv2.INTER_AREA)
        
        alpha_s = filter_img[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(0, 3):
            frame[min_y:max_y, min_x:max_x, c] = (alpha_s * filter_img[:, :, c] + alpha_l * frame[min_y:max_y, min_x:max_x, c])
    
    return frame


# Load the filter (assuming a similar function already exists in the script)
def load_filter(filter_key):
    filter_config = filters_config[filter_key][0]
    filter_img = Image.open(filter_config['path'])

    if filter_config['animated']:
        frames = []
        try:
            while True:
                frame = filter_img.copy().convert('RGBA')
                frames.append(frame)
                filter_img.seek(len(frames))
        except EOFError:
            pass
        filter_img = frames
    else:
        filter_img = [filter_img.convert('RGBA')]
    
    filter_runtime = {
        'image': filter_img,
        'frame_index': 0,
        'has_alpha': filter_config['has_alpha'],
        'morph': filter_config['morph'],
        'animated': filter_config['animated']
    }
    return [filter_runtime]  # Return as list to match expected structure

# Main function or code that starts the process
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    iter_filter_keys = iter(filters_config.keys())
    multi_filter_runtime = load_filter(next(iter_filter_keys))
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        points2 = getLandmarks(frame)
        if points2 is None:
            continue

        if VISUALIZE_FACE_POINTS:
            for idx, point in enumerate(points2):
                cv2.circle(frame, point, 2, (255, 0, 0), -1)
                cv2.putText(frame, str(idx), point, cv2.FONT_HERSHEY_SIMPLEX, .3, (255, 255, 255), 1)
            cv2.imshow("landmarks", frame)

        for idx, filter_runtime in enumerate(multi_filter_runtime):
            if filter_runtime['has_alpha']:
                frame = apply_filter_with_alpha(frame, points2, filter_runtime)
            else:
                frame = apply_filter(frame, points2, filter_runtime)

        cv2.putText(frame, "Press F to change filters", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 1)
        cv2.imshow("Face Filter", frame)

        keypressed = cv2.waitKey(1) & 0xFF
        if keypressed == 27:
            break
        elif keypressed == ord('f'):
            try:
                multi_filter_runtime = load_filter(next(iter_filter_keys))
            except StopIteration:
                iter_filter_keys = iter(filters_config.keys())
                multi_filter_runtime = load_filter(next(iter_filter_keys))

        count += 1

    cap.release()
    cv2.destroyAllWindows()
