import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from pathlib import Path
import pandas as pd
from torchvision import transforms
import mlflow
import json

from custom_dataset import emotionDataset
from custom_model import emotionModel
from engine import Evaluator

log_dir = Path("../logs")
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
        
        with open("../model/normalization_stats.json") as file:
            data = json.load(file)
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[data["Mean"]], std=[data["Standard Deviation"]])
        ])
        
        test_dataset = emotionDataset(test_dataframe, transform=transform)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.debug(f"Evaluating on device: {device}")
        
        num_classes = 3

        class_weights_path = Path("../model/class_weights_stats.json")
        class_weights_data = json.loads(class_weights_path.read_text())
        if class_weights_data is not None:
            class_weights_tensor = torch.tensor(class_weights_data, dtype=torch.float32).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        else:
            criterion = nn.CrossEntropyLoss()
        
        model_path = Path("../model/best_emotion_model.pth")
        best_model = emotionModel(num_classes=num_classes)
        
        best_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        logger.debug(f"Loaded model weights from {model_path}")
        
        evaluator = Evaluator(
            model=best_model,
            test_dataloader=test_dataloader,
            criterion=criterion,
            device=device,
            logger=logger
        )
        
        # Run the evaluation loop
        avg_test_loss, test_acc, macro_f1, class_report = evaluator.evaluate()

        run_info_path = Path("../model/run_info.json")
        if run_info_path.exists():
            run_id = json.loads(run_info_path.read_text())["run_id"]
            
            PROJECT_ROOT = Path(__file__).resolve().parent.parent
            mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT}/mlflow.db")
            
            with mlflow.start_run(run_id=run_id):
                mlflow.log_metrics({
                    "test_loss": avg_test_loss,
                    "test_acc": test_acc,
                    "test_f1": macro_f1,
                })
                mlflow.log_text(class_report, "classification_report.txt")
            logger.debug(f"Logged test metrics to MLflow run {run_id}")
        else:
            logger.error(
                "No run_info.json found next to the model checkpoint — "
                "test metrics were NOT logged to MLflow. Run model_training.py "
                "first so a run_id exists to attach this evaluation to."
            )

        reports_dir = Path("../reports")
        reports_dir.mkdir(exist_ok=True)
        eval_metrics = {
            "test_loss": round(avg_test_loss, 4),
            "test_acc": round(test_acc, 4),
            "test_f1": round(macro_f1, 4),
        }
        (reports_dir / "eval_metrics.json").write_text(json.dumps(eval_metrics, indent=2))
        logger.debug("Saved eval metrics to reports/eval_metrics.json")

        logger.debug("Evaluation Pipeline Completed Successfully.")
        
    except Exception as e:
        logger.error(f"Evaluation Pipeline failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()