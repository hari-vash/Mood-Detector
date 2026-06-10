import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from torchvision import transforms

from custom_dataset import emotionDataset
from custom_model import emotionModel
from engine import Trainer
from utils import calculate_weights,mean_std_calculator

log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("model_training")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG) 

log_file_path = log_dir / "model_training.log"
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def main():
    try:
        logger.debug("Initializing Model Training Pipeline...")
        
        # 1. Load Data
        data_dir = Path("../data/processed")
        train_dataframe = pd.read_pickle(data_dir / "train.pkl")
        validation_dataframe = pd.read_pickle(data_dir / "validation.pkl")
        logger.debug("Loaded Processed Dataframes")
        
        # 2. Compute Mean/Std
        train_mean, train_std = mean_std_calculator(train_dataframe)
        logger.debug(f"Training Mean: {train_mean:.4f}, Std: {train_std:.4f}")
        
        # 3. Define Transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[train_mean], std=[train_std])
        ])
        logger.debug("Transformations Defined")
        
        # 4. Create Datasets and DataLoaders
        train_dataset = emotionDataset(train_dataframe, transform=transform)
        validation_dataset = emotionDataset(validation_dataframe, transform=transform)
        
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
        validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
        logger.debug("DataLoaders Successfully Created")
        
        # 5. Hardware Setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # 6. Loss Function Weights
        num_classes = 3
        total_samples = len(train_dataset)
        samples_in_class = train_dataframe["emotion"].value_counts().sort_index().values
        
        class_weights = calculate_weights(total_samples, samples_in_class, num_classes)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        logger.debug(f"Class Weights Applied: {class_weights}")
        
        # 7. Model, Loss, Optimizer
        model = emotionModel(num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
        logger.debug("Model, Optimizer, and Loss Function initialized")
        
        # 8. Train
        trainer = Trainer(
            epochs=75,
            model=model,
            train_dataloader=train_dataloader,
            validation_dataloader=validation_dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            save_dir="../model",
            logger=logger
        )
        
        trainer.train(log_interval=5)
        logger.debug("Training Pipeline Completed Successfully.")
        
    except Exception as e:
        # exc_info=True prints the actual traceback in the logs, saving you hours of debugging
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()