import os
import shutil
from pathlib import Path
import random
import cv2 as cv

directory_names = {"angry": 0, "fear": 1, "happy": 2, "neutral": 3, "sad": 4, "shock": 5}
dir_path_old = Path(".\\mood-image-dataset\\images\\")
dir_path_new = Path(".\\mood-drawing-dataset\\drawings\\")

# Create new directories if they don't exist
os.makedirs(dir_path_new.parent, exist_ok=True)
os.makedirs(dir_path_new, exist_ok=True)

def dataset_format(directory_names, selected_dir, destination_dir, state):
    source_dir = selected_dir / state
    new_dir = destination_dir / state
    os.makedirs(new_dir, exist_ok=True)

    for i in directory_names.values():
        os.makedirs(new_dir / str(i), exist_ok=True)
    
    for key in directory_names:
        key_dir = source_dir / key
        value_dir = new_dir / str(directory_names[key])
        
        all_files = os.listdir(key_dir)
        selected_files = random.sample(all_files, 300)

        for file in selected_files:
            img_path = key_dir / file
            img = cv.imread(str(img_path))
            if img is None:
                continue
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            inverted = 255 - gray
            blurred = cv.GaussianBlur(inverted, (21, 21), 0)
            drawing = cv.divide(gray, 255 - blurred, scale=256)

            drawing_path = value_dir / file
            cv.imwrite(str(drawing_path), drawing)
            shutil.move(str(img_path), str(drawing_path))

state = ['train', 'validation','final test']
for i in state:
    dataset_format(directory_names, dir_path_old, dir_path_new, i)

print("Dataset Formatted Successfully")
