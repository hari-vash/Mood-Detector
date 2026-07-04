import torch
import numpy as np
from sklearn.metrics import f1_score,classification_report
from pathlib import Path
import logging

class Trainer:
    def __init__(self, epochs, model, train_dataloader, validation_dataloader, optimizer, criterion, device, save_dir="../model", logger=None):
        self.device = device
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = epochs
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logger or logging.getLogger(__name__)
        
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], "validation_F1_score": []}
        self.best_val_f1 = 0.0
        
    def _train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct_preds = 0
        total_samples = len(self.train_dataloader.dataset)
        
        for images, labels in self.train_dataloader:       
            images, labels = images.to(self.device), labels.to(self.device)
    
            self.optimizer.zero_grad()
            output = self.model(images)
            loss = self.criterion(output, labels) 
    
            loss.backward()
            self.optimizer.step()
    
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(output, 1)
            correct_preds += (preds == labels).sum().item()
            
        return running_loss / total_samples, (correct_preds / total_samples) * 100

    def _validate_epoch(self):
        self.model.eval()
        running_loss = 0.0
        correct_preds = 0
        total_samples = len(self.validation_dataloader.dataset)
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for val_images, val_labels in self.validation_dataloader:
                val_images, val_labels = val_images.to(self.device), val_labels.to(self.device)
                
                val_output = self.model(val_images)
                val_loss = self.criterion(val_output, val_labels)
                
                running_loss += val_loss.item() * val_images.size(0)
                
                _, val_preds = torch.max(val_output, 1)
                correct_preds += (val_preds == val_labels).sum().item()

                all_preds.append(val_preds)
                all_labels.append(val_labels)
                
        all_preds = torch.cat(all_preds).cpu().numpy()
        all_labels = torch.cat(all_labels).cpu().numpy()
        
        epoch_loss = running_loss / total_samples
        epoch_acc = (correct_preds / total_samples) * 100
        epoch_f1 = f1_score(all_labels, all_preds, average="macro")
        
        return epoch_loss, epoch_acc, epoch_f1

    def train(self, log_interval=5):
        """
        Args:
            log_interval (int): How often to log the epoch metrics. 
                                Default is 5 (logs epochs 5, 10, 15...).
        """
        self.logger.info("Starting training loop...")
        
        for epoch in range(self.epochs):
            train_loss, train_acc = self._train_epoch()
            val_loss, val_acc, val_f1 = self._validate_epoch()
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['validation_F1_score'].append(val_f1)
            
            is_log_epoch = (epoch + 1) % log_interval == 0
            is_first_or_last = (epoch == 0) or (epoch == self.epochs - 1)
            
            if is_log_epoch or is_first_or_last:
                self.logger.info(
                    f"Epoch [{epoch + 1}/{self.epochs}] | "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.4f}"
                )

            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                save_path = self.save_dir / "best_emotion_model.pth"
                torch.save(self.model.state_dict(), save_path)  
                self.logger.info(f"New best F1 score ({val_f1:.4f}) achieved. Saved model.")
                
        self.logger.info("Training complete.")
        return self.history
    
    
class Evaluator:
    def __init__(self, model, test_dataloader, criterion, device, logger=None):
        self.device = device
        self.model = model.to(self.device)
        self.test_dataloader = test_dataloader
        self.criterion = criterion
        self.logger = logger or logging.getLogger(__name__)
        
    def evaluate(self):
        """Evaluates the model on unseen test data and logs the performance metrics."""
        self.logger.info("Evaluating Model on the unseen test data...")
        self.model.eval()
        
        test_loss = 0.0
        correct_test = 0
        total_samples = len(self.test_dataloader.dataset)
        
        all_test_preds = []
        all_test_labels = []
        
        with torch.no_grad():
            for test_images, test_labels in self.test_dataloader:
                test_images, test_labels = test_images.to(self.device), test_labels.to(self.device)
                
                test_outputs = self.model(test_images)
                loss = self.criterion(test_outputs, test_labels)
                
                test_loss += loss.item() * test_images.size(0)
                
                _, preds = torch.max(test_outputs, 1)
                correct_test += (preds == test_labels).sum().item()
                
                all_test_preds.append(preds)
                all_test_labels.append(test_labels)

        final_preds = torch.cat(all_test_preds).cpu().numpy()
        final_labels = torch.cat(all_test_labels).cpu().numpy()
        
        avg_test_loss = test_loss / total_samples
        test_acc = (correct_test / total_samples) * 100
        macro_f1 = f1_score(final_labels, final_preds, average="macro")
        
        self.logger.info(f"Test Results | Loss: {avg_test_loss:.4f} | Accuracy: {test_acc:.2f}% | Macro F1: {macro_f1:.4f}")
        
        class_report = classification_report(final_labels, final_preds, digits=4)
        self.logger.info(f"\nDetailed Classification Report:\n{class_report}")
        
        return avg_test_loss, test_acc, macro_f1, class_report