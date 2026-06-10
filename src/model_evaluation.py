import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from pathlib import Path
import pandas as pd
from torchvision import transforms

from custom_dataset import emotionDataset
from custom_model import emotionModel
from engine import Evaluator
from utils import mean_std_calculator, calculate_weights

log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

# Terminal gets clean INFO logs, file gets everything
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO) 

log_file_path = log_dir / "model_evaluation.log"
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def main():
    try:
        logger.debug("Initializing Model Evaluation Pipeline...")
        
        data_dir = Path("../data/processed")
        test_dataframe = pd.read_pickle(data_dir / "test.pkl")
        
        train_dataframe = pd.read_pickle(data_dir / "train.pkl")
        train_mean, train_std = mean_std_calculator(train_dataframe)
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[train_mean], std=[train_std])
        ])
        
        test_dataset = emotionDataset(test_dataframe, transform=transform)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.debug(f"Evaluating on device: {device}")
        
        num_classes = 3
        total_train_samples = len(train_dataframe)
        train_samples_in_class = train_dataframe["emotion"].value_counts().sort_index().values
            
        class_weights = calculate_weights(total_train_samples, train_samples_in_class, num_classes)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        
        model_path = Path("../model/best_emotion_model.pth")
        best_model = emotionModel(num_classes=num_classes)
        
        best_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        logger.debug(f"Loaded model weights from {model_path}")
        
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        
        evaluator = Evaluator(
            model=best_model,
            test_dataloader=test_dataloader,
            criterion=criterion,
            device=device,
            logger=logger
        )
        
        # Run the evaluation loop
        evaluator.evaluate()
        logger.debug("Evaluation Pipeline Completed Successfully.")
        
    except Exception as e:
        logger.error(f"Evaluation Pipeline failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()