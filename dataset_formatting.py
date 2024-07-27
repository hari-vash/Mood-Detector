import os
import shutil
import random

# Function to reduce the number of files in a directory to a target size
def reduce_files(directory, target_size):
    files = os.listdir(directory)
    if len(files) > target_size:
        files_to_remove = random.sample(files, len(files) - target_size)
        for file in files_to_remove:
            os.remove(os.path.join(directory, file))

# Function for clubbing the directories and formatting the 'train' dataset
def format(base_dir,state,special_emotions,emotions):

    # Combine disgust and surprise inot a new directory called shock
    shock_dir = os.path.join(base_dir,state,'shock')
    os.makedirs(shock_dir,exist_ok=True)
    for emotion in special_emotions:
        emotion_dir = os.path.join(base_dir,state,emotion)
        files = os.listdir(emotion_dir)
        for file in files:
            shutil.move(os.path.join(emotion_dir,file),shock_dir)
    
    # Reducing files in the 'train' directory
    if state=='train':
        for emotion in emotions:
            emotion_dir = os.path.join(base_dir,state,emotion)
            reduce_files(emotion_dir,target_size)
        reduce_files(shock_dir,target_size)
    
    # Clean up old disgust and surprise directories
    for emotion in special_emotions:
        emotion_dir = os.path.join(base_dir,state,emotion)
        os.rmdir(emotion_dir)

def rename(base_dir,state,directory_names):
    for emotion in directory_names:
        old_name = os.path.join(base_dir,state,emotion)
        new_name = os.path.join(base_dir,state,str(directory_names[emotion]))
        os.rename(old_name,new_name)

# Define the paths to the directories
base_dir = "./fer2013/"
emotions = ['angry', 'fear', 'happy', 'neutral', 'sad']
special_emotions = ['disgust', 'surprise']
target_size = 3600
states = ['train','test']
directory_names = {"angry": 0, "fear": 1, "happy": 2, "neutral": 3, "sad": 4, "shock": 5}

for state in states:
    format(base_dir,state,special_emotions,emotions)    
    rename(base_dir,state,directory_names)
print("Dataset formatted successfully!")
