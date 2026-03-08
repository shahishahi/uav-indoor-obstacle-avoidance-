import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import json
from datetime import datetime
import logging
import sys

# Assuming these are correctly implemented in your local files:
try:
    from bc_model import create_model # Use the factory function
    from il_dataloader import load_dataset
except ImportError as e:
    print(f"Failed to import necessary modules: {e}")
    print("Ensure 'bc_model.py' (with 'create_model' function) and 'il_dataloader.py' (with 'load_dataset' function) are in the Python path.")
    sys.exit(1)

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Default level
stream_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
if not logger.handlers: # Avoid adding multiple handlers if script is re-run in an interactive session
    logger.addHandler(stream_handler)


class BCTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() and not config.get('no_cuda', False) else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.output_dir = config['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)
        
        with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2, sort_keys=True)
        logger.info(f"Configuration saved to {os.path.join(self.output_dir, 'config.json')}")
        
        seed = config.get('seed', 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)
        logger.info(f"Set random seed to {seed}")
        
        self.model = create_model(
            model_type=config.get('model_type', 'full'),
            image_channels=config.get('image_channels', 1),
            velocity_dim=config.get('velocity_dim', 4),
            output_dim=config.get('output_dim', 3),
            dropout_rate=config.get('dropout_rate', 0.2)
        ).to(self.device)
        
        logger.info(f"Model instantiation successful.")
        if hasattr(self.model, 'get_model_info'):
            try:
                info = self.model.get_model_info()
                logger.info("Model Information from get_model_info():")
                for key, value in info.items():
                    logger.info(f"  {key}: {value}")
            except Exception as e:
                logger.warning(f"Could not get model info via get_model_info(): {e}")
                total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                logger.info(f"  Trainable Parameters (fallback): {total_params:,}")
        else:
            total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(f"  Trainable Parameters: {total_params:,}")
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.get('scheduler_factor', 0.5),
            patience=config.get('scheduler_patience', 5)
        )
        
        self.criterion = nn.MSELoss()
        
        self.train_losses = []
        self.val_losses = []
        self.val_maes = []
        self.learning_rates = []
        
        self.best_val_loss = float('inf')
        self.best_model_path = os.path.join(self.output_dir, 'best_model.pt')
        self.final_model_path = os.path.join(self.output_dir, 'final_model.pt')

    def load_data(self):
        logger.info(f"Loading dataset from: {self.config['dataset_dir']}")
        crop_size_config = self.config['image_size'] # e.g., [60, 90]
        # load_dataset expects crop_size as (height, width)
        crop_size_tuple = tuple(crop_size_config) if isinstance(crop_size_config, list) else crop_size_config

        # MODIFIED HERE: Expect load_dataset to return 2 values
        self.train_loader, self.val_loader = load_dataset(
            dataset_dir=self.config['dataset_dir'],
            crop_size=crop_size_tuple, 
            val_split=self.config['val_split'],
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers'],
            seed=self.config.get('seed', 42),
            max_depth=self.config.get('max_depth', 10.0)
        )
        
        # Access datasets from loaders
        train_dataset = self.train_loader.dataset
        val_dataset = self.val_loader.dataset
        
        logger.info(f"Training samples: {len(train_dataset):,}, Batches: {len(self.train_loader):,}")
        logger.info(f"Validation samples: {len(val_dataset):,}, Batches: {len(self.val_loader):,}")

    def _process_batch_data(self, batch_data):
        if len(batch_data) == 3:
            depth_img, vel_input, labels = batch_data
        elif len(batch_data) == 4: 
            depth_img, vel_input, labels, _ = batch_data # Ignore metadata
        else:
            raise ValueError(f"Unexpected number of items in batch: {len(batch_data)}. Expected 3 or 4.")
        return depth_img, vel_input, labels

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch_idx, batch_data in enumerate(self.train_loader):
            depth_img, vel_input, labels = self._process_batch_data(batch_data)
            
            depth_img = depth_img.to(self.device, non_blocking=True)
            vel_input = vel_input.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            labels = torch.clamp(labels, -self.config['max_velocity'], self.config['max_velocity'])
            
            self.optimizer.zero_grad()
            predictions = self.model(depth_img, vel_input)
            loss = self.criterion(predictions, labels)
            loss.backward()
            
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % self.config.get('log_interval', 50) == 0:
                logger.info(f'  Train Batch [{batch_idx+1}/{len(self.train_loader)}]: Loss = {loss.item():.6f}')
        
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0
        all_predictions_list, all_labels_list = [], []
        
        with torch.no_grad():
            for batch_data in self.val_loader:
                depth_img, vel_input, labels = self._process_batch_data(batch_data)

                depth_img = depth_img.to(self.device, non_blocking=True)
                vel_input = vel_input.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                labels = torch.clamp(labels, -self.config['max_velocity'], self.config['max_velocity'])
                
                predictions = self.model(depth_img, vel_input)
                loss = self.criterion(predictions, labels)
                total_loss += loss.item()
                
                all_predictions_list.append(predictions.cpu())
                all_labels_list.append(labels.cpu())
        
        avg_loss = total_loss / len(self.val_loader)
        all_predictions = torch.cat(all_predictions_list, dim=0)
        all_labels = torch.cat(all_labels_list, dim=0)
        
        mae = torch.mean(torch.abs(all_predictions - all_labels)).item()
        return avg_loss, mae

    def save_model_checkpoint(self, filepath, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_type': self.config.get('model_type', 'unknown'),
            'model_class': self.model.__class__.__name__,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config_snapshot': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_maes': self.val_maes,
            'learning_rates': self.learning_rates,
            'best_val_loss': self.best_val_loss
        }
        torch.save(checkpoint, filepath)
        if is_best:
            logger.info(f"New best model saved: {filepath} (Val Loss: {self.best_val_loss:.6f})")

    def plot_training_history(self):
        if not self.train_losses or not self.val_losses:
            logger.warning("No training history to plot.")
            return

        epochs_ran = range(1, len(self.train_losses) + 1)
        plt.figure(figsize=(18, 6))
        
        plt.subplot(1, 3, 1)
        plt.plot(epochs_ran, self.train_losses, label='Train Loss', alpha=0.8)
        plt.plot(epochs_ran, self.val_losses, label='Val Loss', alpha=0.8)
        if self.val_maes:
             plt.plot(epochs_ran, self.val_maes, label='Val MAE', linestyle='--', alpha=0.7)
        plt.xlabel('Epoch'); plt.ylabel('Metric Value'); plt.title('Training & Validation Metrics')
        plt.legend(); plt.grid(True, linestyle=':', alpha=0.6)
        
        plt.subplot(1, 3, 2)
        if self.learning_rates:
            plt.plot(epochs_ran, self.learning_rates, alpha=0.8, marker='.', linestyle='-')
            plt.xlabel('Epoch'); plt.ylabel('Learning Rate'); plt.title('Learning Rate Schedule')
            plt.yscale('log'); plt.grid(True, linestyle=':', alpha=0.6)
        
        plt.subplot(1, 3, 3)
        if len(self.train_losses) > 1 and len(self.val_losses) > 1:
            loss_diff = np.array(self.val_losses) - np.array(self.train_losses)
            plt.plot(epochs_ran, loss_diff, label='Val Loss - Train Loss', alpha=0.8)
            plt.xlabel('Epoch'); plt.ylabel('Difference'); plt.title('Overfitting Monitor')
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
            plt.legend(); plt.grid(True, linestyle=':', alpha=0.6)
        
        plt.tight_layout(pad=2.0)
        plot_path = os.path.join(self.output_dir, 'training_history.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        logger.info(f"Training history plot saved to {plot_path}")
        plt.close()

    def train(self):
        logger.info("Starting training process...")
        logger.info(f"Full configuration: {json.dumps(self.config, indent=2, sort_keys=True)}")
        logger.info(f"Targeting {self.config['epochs']} epochs.")
        
        try:
            self.load_data()
        except Exception as e:
            logger.error(f"Fatal error during data loading: {e}", exc_info=True)
            # Optionally, re-raise or handle more gracefully if needed
            # For now, this will stop the training, which is appropriate
            # if data loading fails.
            return # Exit train method if data loading fails

        last_completed_epoch = -1
        try:
            for epoch in range(self.config['epochs']):
                epoch_start_time = datetime.now()
                current_lr = self.optimizer.param_groups[0]['lr']
                
                train_loss = self.train_epoch()
                val_loss, val_mae = self.validate()
                
                self.scheduler.step(val_loss)
                new_lr = self.optimizer.param_groups[0]['lr']
                
                if new_lr < current_lr:
                    logger.info(f"Epoch {epoch+1}: ReduceLROnPlateau reducing learning rate from {current_lr:.2e} to {new_lr:.2e}.")
                
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                self.val_maes.append(val_mae)
                self.learning_rates.append(current_lr)
                
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    self.save_model_checkpoint(self.best_model_path, epoch, is_best=True)
                
                if (epoch + 1) % self.config.get('save_interval', 10) == 0 or (epoch + 1) == self.config['epochs']:
                    checkpoint_path = os.path.join(self.output_dir, f'checkpoint_epoch_{epoch+1}.pt')
                    self.save_model_checkpoint(checkpoint_path, epoch)
                
                epoch_time = (datetime.now() - epoch_start_time).total_seconds()
                log_msg = (f"Epoch {epoch+1}/{self.config['epochs']} ({epoch_time:.1f}s) | "
                           f"LR: {current_lr:.2e} | Train Loss: {train_loss:.5f} | "
                           f"Val Loss: {val_loss:.5f} | Val MAE: {val_mae:.5f}{' (*Best*)' if is_best else ''}")
                logger.info(log_msg)
                
                if new_lr < self.config.get('min_lr', 1e-7) and epoch < self.config['epochs'] - 1 :
                    logger.info(f"Learning rate ({new_lr:.2e}) below min_lr. Stopping early.")
                    break
                last_completed_epoch = epoch # Mark epoch as completed
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user (KeyboardInterrupt).")
        except Exception as e:
            logger.error(f"An error occurred during training at epoch {last_completed_epoch+1}: {e}", exc_info=True)
        finally:
            logger.info(f"Saving final model state after {last_completed_epoch+1} completed epochs.")
            if hasattr(self, 'model') and self.model is not None: # Ensure model exists before saving
                 self.save_model_checkpoint(self.final_model_path, last_completed_epoch)
            
            self.plot_training_history()
            logger.info("Training process finished.")
            if self.best_val_loss != float('inf'):
                 logger.info(f"Best validation loss achieved: {self.best_val_loss:.6f}")
            logger.info(f"Output and models saved in: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train Behavioral Cloning Model', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, help='Path to JSON config file')
    parser.add_argument('--dataset_dir', type=str, help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, help='Output directory for this run')
    parser.add_argument('--model_type', type=str, choices=['full', 'lightweight'], help='Model architecture type')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--learning_rate', type=float, help='Initial learning rate')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--num_workers', type=int, help='Number of DataLoader workers')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA even if available')
    parser.add_argument('--dropout_rate', type=float, help='Dropout rate for MLP layers')
    parser.add_argument('--image_channels', type=int, help='Number of image channels (e.g., 1 for depth, 3 for RGB)')
    parser.add_argument('--output_dim', type=int, help='Dimension of the output actions')


    args = parser.parse_args()
    
    config = {
        'dataset_dir': '/home/shahi/apf_logs/il_dataset/apf_depth_vel', 
        'output_dir_base': './bc_training',
        'output_dir': None, 
        'epochs': 50,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'model_type': 'full',
        'image_size': [60, 90], 
        'image_channels': 1,    
        'velocity_dim': 4,
        'output_dim': 3,        
        'val_split': 0.2,
        'num_workers': min(4, (os.cpu_count() or 1) // 2 if (os.cpu_count() or 0) > 1 else 0),
        'max_velocity': 2.0,
        'weight_decay': 1e-5,
        'dropout_rate': 0.2,    
        'grad_clip': 1.0,
        'scheduler_factor': 0.5,
        'scheduler_patience': 5,
        'save_interval': 10,
        'log_interval': 50,
        'min_lr': 1e-7,
        'seed': 42,
        'max_depth': 10.0,      
        'no_cuda': False
    }
    
    if args.config and os.path.exists(args.config):
        logger.info(f"Loading configuration from {args.config}")
        with open(args.config, 'r') as f:
            try:
                file_config = json.load(f)
                config.update(file_config)
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON from config file {args.config}: {e}")
                sys.exit(1)
    
    for arg_name, arg_value in vars(args).items():
        if arg_value is not None: 
            config[arg_name] = arg_value

    if config['output_dir'] is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f"{config['model_type']}_lr{config['learning_rate']}_seed{config['seed']}_{timestamp}"
        config['output_dir'] = os.path.join(config['output_dir_base'], run_name)
    
    os.makedirs(config['output_dir'], exist_ok=True)
    logger.info(f"Run output will be saved to: {config['output_dir']}")
    
    if not config.get('dataset_dir') or not os.path.exists(config.get('dataset_dir', '')):
        logger.error(f"Dataset directory not found or not specified: {config.get('dataset_dir')}")
        sys.exit(1)
        
    try:
        trainer = BCTrainer(config)
        trainer.train()
    except Exception as e:
        logger.critical(f"A critical error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()