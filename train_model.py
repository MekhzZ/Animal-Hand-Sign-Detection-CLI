import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
import joblib
import os

current_dir = os.getcwd()

# Load the hand landmarks from the .npy files
landmarks_bull = np.load(os.path.join(current_dir,'npy_files/bull.npy'))
landmarks_cow = np.load(os.path.join(current_dir,'npy_files/cow.npy'))
landmarks_ok = np.load(os.path.join(current_dir,'npy_files/ok.npy'))
landmarks_tiger = np.load(os.path.join(current_dir,'npy_files/tiger.npy'))



# Reshape the landmarks if necessary
# Assuming each landmark set has shape (num_samples, num_landmarks, num_coordinates_per_landmark)
# Flatten the landmarks to shape (num_samples, num_landmarks * num_coordinates_per_landmark)
landmarks_bull = landmarks_bull.reshape(landmarks_bull.shape[0],-1)
landmarks_cow = landmarks_cow.reshape(landmarks_cow.shape[0],-1)
landmarks_ok = landmarks_ok.reshape(landmarks_ok.shape[0],-1)
landmarks_tiger = landmarks_tiger.reshape(landmarks_tiger.shape[0],-1)

# Create labels array with corresponding labels for each set of landmarks
labels_bull = np.array(['bull'] * landmarks_bull.shape[0])
labels_cow = np.array(['cow'] * landmarks_cow.shape[0])
labels_ok = np.array(['ok'] * landmarks_ok.shape[0])
labels_tiger = np.array(['tiger'] * landmarks_tiger.shape[0])


# Combine the landmarks and labels separately
landmarks = np.concatenate((landmarks_bull, landmarks_cow,landmarks_ok, landmarks_tiger), axis=0)
labels = np.concatenate((labels_bull, labels_cow, labels_ok, labels_tiger), axis=0)

# Encode the labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(landmarks, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the SVM classifier
svm_classifier = SVC(kernel='linear', probability=True)
svm_classifier.fit(X_train, y_train)

# Evaluate the classifier
accuracy = svm_classifier.score(X_test, y_test)
print(f"Accuracy: {accuracy}")


# Save the model to a file
joblib.dump(svm_classifier, (os.path.join(current_dir,'exports/svm_animal_sign_model.pkl')))

# Save the label encoder and scaler as well
joblib.dump(label_encoder, (os.path.join(current_dir,'exports/animal_label_encoder.pkl')))
joblib.dump(scaler, (os.path.join(current_dir,'exports/animal_scaler.pkl')))

print("Model and encoders saved successfully.")
