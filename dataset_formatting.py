import os
import shutil
import random

# Define the paths to the directories
base_dir = "./mood-image-dataset/images/train"
emotions = ['angry', 'fear', 'happy', 'neutral', 'sad']
special_emotions = ['disgust', 'surprise']
target_size = 3072

# Function to reduce the number of files in a directory to a target size
def reduce_files(directory, target_size):
    files = os.listdir(directory)
    if len(files) > target_size:
        files_to_remove = random.sample(files, len(files) - target_size)
        for file in files_to_remove:
            os.remove(os.path.join(directory, file))

# Reduce files in each emotion directory
for emotion in emotions:
    emotion_dir = os.path.join(base_dir, emotion)
    reduce_files(emotion_dir, target_size)

# Combine disgust and surprise into a new directory called shock
shock_dir = os.path.join(base_dir, 'shock')
os.makedirs(shock_dir, exist_ok=True)

for emotion in special_emotions:
    emotion_dir = os.path.join(base_dir, emotion)
    files = os.listdir(emotion_dir)
    for file in files:
        shutil.move(os.path.join(emotion_dir, file), shock_dir)

# Reduce files in the shock directory to the target size
reduce_files(shock_dir, target_size)

# Clean up old disgust and surprise directories
for emotion in special_emotions:
    emotion_dir = os.path.join(base_dir, emotion)
    os.rmdir(emotion_dir)

print("Dataset formatted successfully!")
