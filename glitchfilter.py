import mediapipe as mp
import cv2
import math
import numpy as np
import csv

VISUALIZE_FACE_POINTS = False

# Define the glitch filter configuration
filters_config = {
    'glitch': [{'path': "mnt\data\glitch.gif",  # Update this path
                'anno_path': "mnt/data/facefilterpoints.csv",  # Update this path
                'morph': False, 'animated': True, 'has_alpha': False}],
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
    return None

def load_filter_img(img_path, has_alpha):
    # Read the image
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    alpha = None
    if has_alpha:
        b, g, r, alpha = cv2.split(img)
        img = cv2.merge((b, g, r))

    return img, alpha

def load_landmarks(annotation_file):
    with open(annotation_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        points = {}
        for i, row in enumerate(csv_reader):
            # skip head or empty line if it's there
            try:
                x, y = int(row[0]), int(row[1])
                points[i] = (x, y)
            except ValueError:
                continue
        return points

def load_filter(filter_name="glitch"):
    filters = filters_config[filter_name]
    multi_filter_runtime = []

    for filter in filters:
        temp_dict = {}
        if filter['animated']:
            filter_cap = cv2.VideoCapture(filter['path'])
            temp_dict['cap'] = filter_cap
        else:
            img1, img1_alpha = load_filter_img(filter['path'], filter['has_alpha'])
            temp_dict['img'] = img1
            temp_dict['img_a'] = img1_alpha

        points = load_landmarks(filter['anno_path'])
        temp_dict['points'] = points

        multi_filter_runtime.append(temp_dict)

    return filters, multi_filter_runtime

def apply_filter(frame, points2, filter_runtime):
    ret, filter_frame = filter_runtime['cap'].read()
    if not ret:
        filter_runtime['cap'].set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, filter_frame = filter_runtime['cap'].read()

    # Ensure the filter frame is the same size as the webcam frame
    filter_frame = cv2.resize(filter_frame, (frame.shape[1], frame.shape[0]))

    points1 = list(filter_runtime['points'].values())

    # Transform the filter frame to align with the detected face landmarks
    if len(points1) == len(points2):
        tform, _ = cv2.estimateAffinePartial2D(np.array(points1), np.array(points2))
        transformed_filter_frame = cv2.warpAffine(filter_frame, tform, (frame.shape[1], frame.shape[0]))
        mask = cv2.cvtColor(transformed_filter_frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        # Black-out the area of filter in the frame
        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)

        # Take only the region of the filter from the transformed filter frame
        filter_fg = cv2.bitwise_and(transformed_filter_frame, transformed_filter_frame, mask=mask)

        # Combine the background and the filter foreground
        output = cv2.add(frame_bg, filter_fg)
        return output

    return frame

# Process input from webcam or video file
cap = cv2.VideoCapture(0)

# Some variables
count = 0
isFirstFrame = True
sigma = 50

iter_filter_keys = iter(filters_config.keys())
filters, multi_filter_runtime = load_filter(next(iter_filter_keys))

# The main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break
    else:
        points2 = getLandmarks(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # If face is partially detected
        if not points2 or (len(points2) != 75):
            continue

        ################ Optical Flow and Stabilization Code #####################
        img2Gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if isFirstFrame:
            points2Prev = np.array(points2, np.float32)
            img2GrayPrev = np.copy(img2Gray)
            isFirstFrame = False

        lk_params = dict(winSize=(101, 101), maxLevel=15,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.001))
        points2Next, st, err = cv2.calcOpticalFlowPyrLK(img2GrayPrev, img2Gray, points2Prev,
                                                        np.array(points2, np.float32),
                                                        **lk_params)

        # Final landmark points are a weighted average of detected landmarks and tracked landmarks
        for k in range(0, len(points2)):
            d = cv2.norm(np.array(points2[k]) - points2Next[k])
            alpha = math.exp(-d * d / sigma)
            points2[k] = (1 - alpha) * np.array(points2[k]) + alpha * points2Next[k]
            points2[k] = (int(points2[k][0]), int(points2[k][1]))

        # Update variables for next pass
        points2Prev = np.array(points2, np.float32)
        img2GrayPrev = img2Gray
        ################ End of Optical Flow and Stabilization Code ###############

        if VISUALIZE_FACE_POINTS:
            for idx, point in enumerate(points2):
                cv2.circle(frame, point, 2, (255, 0, 0), -1)
                cv2.putText(frame, str(idx), point, cv2.FONT_HERSHEY_SIMPLEX, .3, (255, 255, 255), 1)
            cv2.imshow("landmarks", frame)

        for idx, filter in enumerate(filters):
            filter_runtime = multi_filter_runtime[idx]
            frame = apply_filter(frame, points2, filter_runtime)

        cv2.putText(frame, "Press F to change filters", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 1)
        cv2.imshow("Face Filter", frame)

        keypressed = cv2.waitKey(1) & 0xFF
        if keypressed == 27:
            break
        # Put next filter if 'f' is pressed
        elif keypressed == ord('f'):
            try:
                filters, multi_filter_runtime = load_filter(next(iter_filter_keys))
            except StopIteration:
                iter_filter_keys = iter(filters_config.keys())
                filters, multi_filter_runtime = load_filter(next(iter_filter_keys))

        count += 1

cap.release()
cv2.destroyAllWindows()
