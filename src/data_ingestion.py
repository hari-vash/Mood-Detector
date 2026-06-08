import os
import logging
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import zipfile

# Ensure "logs" directory exists
log_dir = "logs"
os.makedirs(".//logs", exist_ok=True)

# Configuring logging
logger = logging.getLogger("data_ingestion")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG") 

log_file_path = os.path.join(log_dir,"data_ingestion.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s -%(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)    

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def get_data(target_dir="..//data"):
    """Download the data and store it in a ./data/raw folder"""
    file_path = os.path.join(target_dir,"raw","challenges-in-representation-learning-facial-expression-recognition-challenge.zip")
    try:
        if os.path.exists(file_path):
            logger.debug("Dataset alreayd exists. Skipping download...")
            return file_path
        logger.debug("Dataset not found. Downloading to ../data/raw")
        os.makedirs("..//data",exist_ok=True)
        os.system(
            "kaggle competitions download challenges-in-representation-learning-facial-expression-recognition-challenge -p ..//data//raw"
        )
        logger.debug("Dataset Downloaded")
        return file_path
    
    except Exception as e:
        logger.error("Unexpected Error occured while downloading the data: %s",e)
        raise
    
def unzip_data(file_path,target_dir):
    """Unzip the downloaded data and return the train.csv file"""
    try:
        file_path = os.path.join(target_dir,"raw")
        if os.path.exists(os.path.join(target_dir,"train.csv")):
            logger.debug("Data already extracted.")
            return file_path
        with zipfile.ZipFile(file_path,"r") as zip_ref:
            zip_ref.extractall("..//data//raw")
        logger.debug("Data Extraction Completed.")
        return file_path
    except Exception as e:
        logger.error("Unexpected Error occured while unzipping the data: %s",e)
        raise
    
def preprocess_data(dataframe):
    """Drop and rename classes and data points"""
    try:
        classes_to_drop = [1,2,5,6]
        label_mapping = {0:0, 3:1, 4:2}
        dataframe = dataframe[~dataframe["emotion"].isin(classes_to_drop)]
        dataframe["emotion"] = dataframe["emotion"].map(label_mapping)
        dataframe['pixels'] = dataframe['pixels'].str.split().apply(lambda x: np.array([float(i) for i in x], dtype=np.float32))
        train_dataframe, test_dataframe = train_test_split(dataframe,test_size=0.20,stratify=dataframe["emotion"],random_state=42)
        train_dataframe, validation_dataframe = train_test_split(train_dataframe,test_size=0.20,stratify=train_dataframe["emotion"],random_state=42)
        logger.debug("Data Preprocessing Completed.")
        return train_dataframe,validation_dataframe,test_dataframe
    except Exception as e:
        logger.error("Unexpected Error occured while preprocessing the data: %s",e)
        raise
    
def save_data(target_dir,train_dataframe,validation_dataframe,test_dataframe):
    """Save the newly preprocessed data in ./data/processed"""
    try:
        save_path = os.path.join(target_dir,"processed")
        os.makedirs(save_path,exist_ok=True)
        train_dataframe.to_pickle("..//data//processed//train.pkl")
        validation_dataframe.to_pickle("..//data//processed//validation.pkl")
        test_dataframe.to_pickle("..//data//processed//test.pkl")
        logger.debug("Saved processed data.")
    except Exception as e:
        logger.error("Unexpected Error occured while saving the data: %s",e)
        raise

def main():
    try:
        logger.debug("Initializing Data Ingestion Pipeline")
        target_dir = "..//data"
        data_path = get_data(target_dir)
        unzipped_data = unzip_data(data_path,target_dir)
        dataframe = pd.read_csv(unzipped_data)
        train_dataframe, validation_dataframe, test_dataframe = preprocess_data(dataframe)
        save_data(target_dir,train_dataframe,validation_dataframe,test_dataframe)
        logger.debug("Data Ingestion Pipeline Completed.")
    except Exception as e:
        logger.error("Unexpected Error occured while running Data Ingestion Pipeline: %s",e)
        raise    

if __name__ == "__main__":
    main()