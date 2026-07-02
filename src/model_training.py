import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from torchvision import transforms
import mlflow
import json 

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
        
        # 1. Define Parameters
        params = {"batch_size":32,"epochs":67,"learning_rate":0.001,"weight_decay":1e-3}
        
        # 2. Load Data
        data_dir = Path("data/processed")
        train_dataframe = pd.read_pickle(data_dir / "train.pkl")
        validation_dataframe = pd.read_pickle(data_dir / "validation.pkl")
        logger.debug("Loaded Processed Dataframes")
        
        # 3. Compute Mean/Std
        train_mean, train_std = mean_std_calculator(train_dataframe)
        normalization_stats = {
            "Mean" : train_mean,
            "Standard Deviation": train_std
        }
        logger.debug(f"Training Mean: {train_mean:.4f}, Std: {train_std:.4f}")
        
        # 4. Define Transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[train_mean], std=[train_std])
        ])
        logger.debug("Transformations Defined")
        
        # 5. Create Datasets and DataLoaders
        train_dataset = emotionDataset(train_dataframe, transform=transform)
        validation_dataset = emotionDataset(validation_dataframe, transform=transform)
        
        train_dataloader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True, drop_last=True)
        validation_dataloader = DataLoader(validation_dataset, batch_size=params["batch_size"], shuffle=False)
        logger.debug("DataLoaders Successfully Created")
        
        # 6. Hardware Setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # 7. Loss Function Weights
        num_classes = 3
        total_samples = len(train_dataset)
        samples_in_class = train_dataframe["emotion"].value_counts().sort_index().values
        
        class_weights = calculate_weights(total_samples, samples_in_class, num_classes)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        logger.debug(f"Class Weights Applied: {class_weights}")
        
        # 8. Model, Loss, Optimizer
        model = emotionModel(num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
        logger.debug("Model, Optimizer, and Loss Function initialized")
        
        PROJECT_ROOT = Path(__file__).resolve().parent.parent
        mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT}/mlflow.db")
        mlflow.set_experiment("mood-detector")
        with mlflow.start_run() as run:
            
            mlflow.log_params({
                "epochs":params["epochs"],
                "batch_size":params["batch_size"],
                "learning_rate":params["learning_rate"],
                "weight_decay":params["weight_decay"],
                "optimizer":"Adam",
                "num_classes":num_classes,
                "architecture":"emotionModel (residual CNN)"
            })
            
            # 9. Train
            trainer = Trainer(
                epochs=params["epochs"],
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                train_dataloader=train_dataloader,
                validation_dataloader=validation_dataloader,
                device=device,
                save_dir="model",
                logger=logger
            )

            history = trainer.train(log_interval=5)
            
            for epoch, (train_loss,train_acc,val_loss,val_acc,f1_score) in enumerate(zip(history["train_loss"],history["train_acc"],history["val_loss"],history["val_acc"],history["validation_F1_score"])):
                mlflow.log_metrics(
                    {
                        "Train Loss":train_loss,
                        "Train Accuracy":train_acc,
                        "Validation Loss":val_loss,
                        "Validation Accuracy":val_acc,
                        "Validation F1 Score":f1_score
                    },
                    step=epoch
                )
                mlflow.log_metric("Best Validation F1 Score",trainer.best_val_f1)
                
            best_model_path = Path("model/best_emotion_model.pth")
            if best_model_path.exists():
                mlflow.log_artifact(str(best_model_path),artifact_path="model")
                
            run_info_path = Path("model/run_info.json")
            run_info_path.write_text(json.dumps({"run_id": run.info.run_id}))
            logger.debug(f"Saved MLflow run_id to {run_info_path}")

            normalization_stats_path = Path("model/normalization_stats.json")
            normalization_stats_path.write_text(json.dumps(normalization_stats))
            logger.debug(f"Saved Normalization Stats to {normalization_stats_path}")
                
            class_weights_path = Path("model/class_weights_stats.json")
            class_weights_path.write_text(json.dumps(class_weights))
            logger.debug(f"Saved Class weights stats to {class_weights_path}")
                
        logger.debug("Training Pipeline Completed Successfully.")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()