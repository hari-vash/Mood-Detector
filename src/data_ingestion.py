import os
import logging
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# Ensure "logs" directory exists inside the project
log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)

# Configuring logging
logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG) 

log_file_path = log_dir / "data_ingestion.log"
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)    

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def get_data(target_dir):
    """Download the data and store it inside the project's data/raw folder"""
    target_dir = Path(target_dir)
    raw_dir = target_dir / "raw"
    zip_file_path = raw_dir / "challenges-in-representation-learning-facial-expression-recognition-challenge.zip"
    
    if zip_file_path.exists():
        logger.debug("Dataset zip found at location. Skipping download...")
        return zip_file_path
        
    logger.debug("Dataset zip not found. Creating directory.")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    logger.debug("Triggering Kaggle API download command...")
    os.system(
        f'kaggle competitions download challenges-in-representation-learning-facial-expression-recognition-challenge -p "{raw_dir.resolve()}"'
    )
    
    if not zip_file_path.exists():
        raise FileNotFoundError(
            f"Kaggle download command executed, but no file appeared at: {zip_file_path.resolve()}. "
            f"Please verify your kaggle.json credentials and make sure you accepted the competition rules online!"
        )
        
    logger.debug("Dataset Downloaded successfully.")
    return zip_file_path
    
def unzip_data(zip_file_path, target_dir):
    """Unzip the downloaded data and return the extracted CSV file path"""
    try:
        zip_file_path = Path(zip_file_path)
        target_dir = Path(target_dir)
        raw_dir = target_dir / "raw"
        
        existing_csvs = list(raw_dir.glob("*.csv"))
        if existing_csvs:
            csv_path = raw_dir / "train.csv"
            logger.debug("Extracted CSV data already exists: %s. Skipping extraction.", csv_path.name)
            return csv_path
            
        logger.debug("Extracting zip archive from %s...", zip_file_path.name)
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(raw_dir)
            
        # Re-check for the CSV file now that it is unzipped
        extracted_csvs = list(raw_dir.glob("*.csv"))
        if not extracted_csvs:
            raise FileNotFoundError(f"No CSV file found in {raw_dir.resolve()} after extraction.")
            
        csv_path = raw_dir / "train.csv"
        logger.debug("Data Extraction Completed. Found data file: %s", csv_path.name)
        return csv_path
        
    except Exception as e:
        logger.error("Unexpected Error occurred while unzipping the data: %s", e)
        raise
    
def preprocess_data(dataframe):
    """Drop and rename classes and data points"""
    try:
        classes_to_drop = [1, 2, 5, 6]
        label_mapping = {0: 0, 3: 1, 4: 2}
        
        dataframe = dataframe[~dataframe["emotion"].isin(classes_to_drop)].copy()
        dataframe["emotion"] = dataframe["emotion"].map(label_mapping)
        
        logger.debug("Converting pixels string column to float numpy arrays...")
        dataframe['pixels'] = dataframe['pixels'].str.split().apply(lambda x: np.array([float(i) for i in x], dtype=np.float32))
        
        train_dataframe, test_dataframe = train_test_split(dataframe, test_size=0.20, stratify=dataframe["emotion"], random_state=42)
        train_dataframe, validation_dataframe = train_test_split(train_dataframe, test_size=0.20, stratify=train_dataframe["emotion"], random_state=42)
        
        logger.debug("Data Preprocessing Completed.")
        return train_dataframe, validation_dataframe, test_dataframe
    except Exception as e:
        logger.error("Unexpected Error occurred while preprocessing the data: %s", e)
        raise
    
def save_data(target_dir, train_dataframe, validation_dataframe, test_dataframe):
    """Save the newly preprocessed data in data/processed"""
    try:
        processed_path = Path(target_dir) / "processed"
        processed_path.mkdir(parents=True, exist_ok=True)
        
        train_dataframe.to_pickle(processed_path / "train.pkl")
        validation_dataframe.to_pickle(processed_path / "validation.pkl")
        test_dataframe.to_pickle(processed_path / "test.pkl")
        
        logger.debug("Saved processed data.")
    except Exception as e:
        logger.error("Unexpected Error occurred while saving the data: %s", e)
        raise

def main():
    try:
        logger.debug("Initializing Data Ingestion Pipeline")
        
        # CHANGED: Use "data" instead of "../data" to keep it INSIDE your project directory
        target_dir = Path("data") 
        
        # 1. Download zip file
        zip_path = get_data(target_dir)
        
        # 2. Extract zip file and find CSV data file
        csv_data_path = unzip_data(zip_path, target_dir)
        
        # 3. Read CSV data
        logger.debug("Reading data.")
        dataframe = pd.read_csv(csv_data_path)
        
        # 4. Process and split
        train_df, val_df, test_df = preprocess_data(dataframe)
        
        # 5. Save splits to disk
        save_data(target_dir, train_df, val_df, test_df)
        
        logger.debug("Data Ingestion Pipeline Completed successfully.")
    except Exception as e:
        logger.error("Unexpected Error occurred while running Data Ingestion Pipeline: %s", e)
        raise    

if __name__ == "__main__":
    main()
