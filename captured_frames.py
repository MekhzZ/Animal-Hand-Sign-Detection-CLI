import cv2
import os

current_dir = os.getcwd()

# Define paths 
# I copied ans pasted each file path one by one , I was thinking of creating loop that will automate every files in one go 
# feel free to Pull requests and contribute 

video_path = os.path.join(current_dir,'video_files/bull/bull.mp4') #copy the video path and paste it here
output_folder = os.path.join('Captured_frames/bull') #copy the output path in this repo you can save output folder in Captured frames


# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get the list of existing .jpg files in the output folder
existing_files = [f for f in os.listdir(output_folder) if f.endswith('.jpg')]

# Determine the highest numbered frame
if existing_files:
    existing_numbers = [int(f.split('_')[1].split('.')[0]) for f in existing_files]
    max_existing_number = max(existing_numbers)
    frame_count = max_existing_number + 1
else:
    frame_count = 0

# Open the video file
cap = cv2.VideoCapture(video_path)

# Capture frames from the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Save frame with the new numbering scheme
    cv2.imwrite(os.path.join(output_folder, f'frame_{frame_count}.jpg'), frame)
    frame_count += 1

# Release the video capture object
cap.release()
