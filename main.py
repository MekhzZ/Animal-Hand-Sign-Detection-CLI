import cv2
import mediapipe as mp
import joblib
import numpy as np
import os

current_dir = os.getcwd()

model = joblib.load('Trained_model/svm_animal_sign_model.pkl')
label_encoder = joblib.load('Trained_model/animal_label_encoder.pkl')
scaler = joblib.load('Trained_model/animal_scaler.pkl')

# Initialize Mediapipe for hand tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Setup webcam capture
cap = cv2.VideoCapture(0)

# Confidence threshold for displaying the animal name
confidence_threshold = 0.80

# Initialize Mediapipe hands model
with mp_hands.Hands(model_complexity=1, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Process the frame to detect hand landmarks
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Extract landmarks as a flat list
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                # Scale the landmarks using the trained scaler
                scaled_landmarks = scaler.transform(np.array(landmarks).reshape(1, -1))

                try:
                    # Get prediction probabilities
                    probabilities = model.predict_proba(scaled_landmarks)[0]
                    predicted_class = np.argmax(probabilities)
                    predicted_proba = probabilities[predicted_class]

                    # Only display animal name if confidence is above the threshold
                    if predicted_proba >= confidence_threshold:
                        animal_name = label_encoder.inverse_transform([predicted_class])[0]
                        cv2.putText(frame, f"Detected: {animal_name} ({predicted_proba:.2f})", (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                    else:
                        animal_name = "Unknown"
                        cv2.putText(frame, f"Confidence too low", (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                except Exception as e:
                    print(f"Prediction error: {e}")
                    animal_name = "Unknown"

                # Draw the landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display the frame with landmarks and predictions
        cv2.imshow('Hand Sign Detection', frame)

        # Break the loop with 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
