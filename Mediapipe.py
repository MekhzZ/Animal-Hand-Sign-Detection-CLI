import mediapipe as mp
import numpy as np
import cv2 
import os

current_dir = os.getcwd()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# I copied ans pasted each file path one by one , I was thinking of creating loop that will automate every files in one go 
# feel free to Pull requests and contribute 

frames_folder = os.path.join(current_dir,'Captured_frames/bull')
landmarks_list = []

for frame_file in os.listdir(frames_folder):
    image = cv2.imread(os.path.join(frames_folder, frame_file))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark]).flatten()
            landmarks_list.append(landmarks)

# Save landmarks to a file for later use
np.save((os.path.join(current_dir,'npy_files/bull.npy')), landmarks_list)
