import opendatasets as od
from pathlib import Path

if Path(".//challenges-in-representation-learning-facial-expression-recognition-challenge").is_file():
    print("Dataset alreay present. Skipping Download...")
else:
    print("Downloading the dataset...")
    od.download("https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data")
