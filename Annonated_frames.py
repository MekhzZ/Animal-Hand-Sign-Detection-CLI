import cv2
import os
import mediapipe as mp
import numpy as np

current_dir = os.getcwd()

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# I copied ans pasted each file path one by one , I was thinking of creating loop that will automate every files in one go 
# feel free to Pull requests and contribute 

landmarks_list = np.load(os.path.join(current_dir,'npy_files/cow.npy'))

frames_folder = os.path.join(current_dir,'Captured_frames/cow')
annotated_folder = os.path.join(current_dir,'Annonated_frames/cow')

if not os.path.exists(annotated_folder):
    os.makedirs(annotated_folder)

for idx, frame_file in enumerate(os.listdir(frames_folder)):
    if not frame_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    frame_path = os.path.join(frames_folder, frame_file)
    image = cv2.imread(frame_path)
    height, width, _ = image.shape

    landmarks = landmarks_list[idx]

    hand_landmarks = landmarks.reshape(-1, 3)
    for landmark in hand_landmarks:
        cx = int(landmark[0] * width)
        cy = int(landmark[1] * height)
        cz = landmark[2]  # Z-coordinates are relative

        cv2.circle(image, (cx, cy), 3, (0, 255, 0), -1)

    annotated_frame_path = os.path.join(annotated_folder, 'annotated_' + frame_file)
    cv2.imwrite(annotated_frame_path, image)

print(f"Annotated frames saved in {annotated_folder}")
