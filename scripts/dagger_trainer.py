#!/usr/bin/env python3

import torch
# import torch.nn as nn # Not directly used
import torch.optim as optim # For type hinting if needed
import numpy as np
import os
import json
import pickle
import shutil
from datetime import datetime
import logging
# from collections import defaultdict # Not used
import csv # For writing CSV
import cv2 # For saving images

from train_bc import BCTrainer # Uses the existing BCTrainer
# --- MODIFIED --- Use the new loader that can handle multiple sources
from il_dataloader import load_dataset_from_sources, logger as il_logger

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
il_logger.setLevel(logging.INFO) # Ensure il_dataloader logs are also visible


class DAggerTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.output_dir = config['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.current_dagger_iteration = 0 # Renamed for clarity
        self.iteration_history = []
        
        self.aggregated_dataset_sources = [] # List of dicts: {'key': str, 'csv_path': str, 'img_dir': str, 'weight': float}
        
        logger.info(f"[DAggerTrainer] Initialized. Output: {self.output_dir}. Device: {self.device}")

    def load_initial_bc_dataset(self): # Renamed for clarity
        initial_data_dir = self.config.get('initial_bc_dataset_dir') # --- MODIFIED --- key name
        if not initial_data_dir or not os.path.exists(initial_data_dir):
            logger.error(f"Initial BC dataset directory not found or not specified: {initial_data_dir}")
            raise FileNotFoundError(f"Initial BC dataset directory not found: {initial_data_dir}")
        
        logger.info(f"[DAggerTrainer] Adding initial BC dataset from: {initial_data_dir}")
        
        # --- MODIFIED --- Standardize how sources are added
        self.aggregated_dataset_sources.append({
            'key': 'initial_bc_data',
            'csv_path': os.path.join(initial_data_dir, 'controls.csv'),
            'img_dir': os.path.join(initial_data_dir, 'images'),
            'type': 'initial_bc',
            'iteration': 0, # Belongs to iteration 0 (pre-DAgger)
            'weight': self.config.get('initial_dataset_weight', 1.0) # Configurable weight
        })
        logger.info(f"[DAggerTrainer] Initial BC dataset source added: {self.aggregated_dataset_sources[-1]}")


    def add_dagger_collected_data(self, dagger_pkl_path, iteration): # Renamed for clarity
        if not os.path.exists(dagger_pkl_path):
            logger.error(f"DAgger collected data .pkl file not found: {dagger_pkl_path}")
            raise FileNotFoundError(f"DAgger .pkl data not found: {dagger_pkl_path}")
        
        logger.info(f"[DAggerTrainer] Adding DAgger iteration {iteration} data from: {dagger_pkl_path}")
        
        try:
            with open(dagger_pkl_path, 'rb') as f:
                dagger_data_content = pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load DAgger .pkl data from {dagger_pkl_path}: {e}", exc_info=True)
            return

        num_points = dagger_data_content.get('collection_info', {}).get('total_datapoints', 0)
        if num_points == 0:
            logger.warning(f"[DAggerTrainer] No datapoints found in DAgger .pkl file: {dagger_pkl_path}. Skipping.")
            return
        logger.info(f"[DAggerTrainer] Loaded {num_points} datapoints from DAgger iteration {iteration} pkl.")
        
        # --- MODIFIED --- Convert DAgger .pkl data to CSV/PNG dataset format
        converted_data_dir_path = self._convert_dagger_pkl_to_csv_png_dataset(dagger_data_content, iteration)
        if not converted_data_dir_path:
            logger.error(f"Failed to convert DAgger pkl for iteration {iteration}. Skipping.")
            return

        weight = self._calculate_iteration_data_weight(iteration) # --- MODIFIED --- Renamed
        
        self.aggregated_dataset_sources.append({
            'key': f'dagger_iter_{iteration}',
            'csv_path': os.path.join(converted_data_dir_path, 'controls.csv'),
            'img_dir': os.path.join(converted_data_dir_path, 'images'),
            'type': 'dagger_collected',
            'iteration': iteration,
            'weight': weight,
            'num_points': num_points
        })
        logger.info(f"[DAggerTrainer] DAgger iteration {iteration} data source added: {self.aggregated_dataset_sources[-1]}")


    # --- MODIFIED --- This function now converts to CSV/PNG
    def _convert_dagger_pkl_to_csv_png_dataset(self, dagger_data_content, iteration):
        """
        Converts DAgger pickle data (containing raw_datapoints) to a standard 
        CSV/PNG dataset format, similar to the initial BC dataset.
        Saves depth images as uint16 PNGs (in mm).
        """
        dataset_output_dir = os.path.join(self.output_dir, f'converted_dagger_dataset_iter_{iteration}')
        images_subdir = os.path.join(dataset_output_dir, 'images')
        csv_filepath = os.path.join(dataset_output_dir, 'controls.csv')

        os.makedirs(images_subdir, exist_ok=True)
        
        datapoints_list = dagger_data_content.get('raw_datapoints', [])
        if not datapoints_list:
            logger.warning(f"[DAggerTrainer] No 'raw_datapoints' in DAgger pkl for iteration {iteration}.")
            return None

        max_depth_val_config = self.config.get('max_depth', 10.0) # Used for clipping before saving as mm

        try:
            with open(csv_filepath, 'w', newline='') as csvfile:
                # Consistent with bag_to_csv.py output and il_dataloader.py input
                fieldnames = ['timestamp', 'vx_input', 'vy_input', 'vz_input', 'yaw_rate', 
                              'vx', 'vy', 'vz', 'image_file']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                num_saved = 0
                for i, dp in enumerate(datapoints_list):
                    # Depth image from pkl is 'depth_image_meters' (float32, meters)
                    depth_meters_np = dp.get('depth_image_meters')
                    if depth_meters_np is None:
                        logger.warning(f"Skipping datapoint {i} in iter {iteration} due to missing 'depth_image_meters'")
                        continue

                    # Convert to uint16, mm for saving as PNG
                    # Clip to max_depth before converting to mm to avoid overflow if values are unexpectedly large
                    depth_meters_clipped = np.clip(depth_meters_np, 0, max_depth_val_config)
                    depth_mm_uint16 = (depth_meters_clipped * 1000).astype(np.uint16)
                    
                    img_filename = f"iter{iteration}_{dp['timestamp']:.6f}_{i}.png"
                    img_save_path = os.path.join(images_subdir, img_filename)
                    
                    if not cv2.imwrite(img_save_path, depth_mm_uint16):
                        logger.error(f"Failed to save image: {img_save_path}")
                        continue # Skip this datapoint if image saving fails

                    # Prepare CSV row
                    # expert_action should be 3D, velocity_input should be 4D
                    expert_act = dp.get('expert_action', np.zeros(3))
                    vel_in = dp.get('velocity_input', np.zeros(4))

                    csv_row = {
                        'timestamp': dp['timestamp'],
                        'vx_input': vel_in[0], 'vy_input': vel_in[1], 
                        'vz_input': vel_in[2], 'yaw_rate': vel_in[3],
                        'vx': expert_act[0], 'vy': expert_act[1], 'vz': expert_act[2],
                        'image_file': img_filename
                    }
                    writer.writerow(csv_row)
                    num_saved += 1
            
            logger.info(f"[DAggerTrainer] Converted {num_saved}/{len(datapoints_list)} DAgger datapoints for iteration {iteration} to CSV/PNG format at: {dataset_output_dir}")
            return dataset_output_dir if num_saved > 0 else None

        except Exception as e:
            logger.error(f"Error converting DAgger pkl for iter {iteration} to CSV/PNG: {e}", exc_info=True)
            return None


    def _calculate_iteration_data_weight(self, iteration): # Renamed
        strategy = self.config.get('dagger_dataset_weighting_strategy', 'recent_bias') # --- MODIFIED --- key name
        
        if strategy == 'equal':
            return 1.0
        elif strategy == 'recent_bias':
            decay = self.config.get('dagger_weight_decay_factor', 0.9) # --- MODIFIED --- key name
            # Weight relative to the current DAgger iteration being processed
            # Iteration 1 gets full weight from its own data, iteration 2 means iter 1 data gets decay^1, etc.
            # This needs to be applied carefully when combining all datasets.
            # For simplicity, this function could just return a base weight, and actual sampling handled by DataLoader.
            # For now, let's assume this weight is used to determine how many times to repeat samples from this iteration's data.
            # A simpler interpretation: max_iterations could be the *current* highest iteration number.
            # If this is iteration `k`, and we are on iteration `N_max`, then weight is decay^(N_max - k)
            # This seems more complex than needed if the dataloader handles weighted sampling.
            # Let's just return a factor based on iteration number for now.
            # Iteration 1: 1.0, Iteration 2: decay, Iteration 3: decay^2 etc.
            return decay ** (iteration -1) if iteration > 0 else 1.0 # Iteration 0 (initial_bc) gets 1.0
        else: # Default to equal
            return 1.0

    def _create_aggregated_dataloader_for_training(self): # Renamed
        logger.info("[DAggerTrainer] Creating aggregated dataloader for training...")
        
        if not self.aggregated_dataset_sources:
            logger.error("[DAggerTrainer] No dataset sources available to create dataloader.")
            raise ValueError("No dataset sources for DAgger training.")

        # --- MODIFIED --- Use the new load_dataset_from_sources
        # Note: The 'weight' in aggregated_dataset_sources is not directly used by load_dataset_from_sources yet.
        # Proper weighting would require a custom sampler in DataLoader or over/undersampling the metadata list
        # before passing to DepthVelDataset. For now, it's a simple concatenation.
        
        # For true weighting, you'd do something like:
        # weighted_meta_data = []
        # for source in self.aggregated_dataset_sources:
        #     # Load source's meta_data
        #     # Repeat samples from this source based on source['weight'] before adding to combined list
        # This is a simplification for now. A WeightedRandomSampler in DataLoader is better.

        train_loader, val_loader = load_dataset_from_sources(
            dataset_sources=self.aggregated_dataset_sources, # Pass the list of source dicts
            crop_size=tuple(self.config.get('image_size', [60, 90])),
            val_split=self.config.get('val_split', 0.2),
            batch_size=self.config.get('batch_size', 32),
            num_workers=self.config.get('num_workers', self.config.get('train_num_workers', 4)), # More specific key
            seed=self.config.get('seed', None), # Allow None for no fixed seed
            max_depth=self.config.get('max_depth', 10.0),
            shuffle_train=self.config.get('shuffle_train_globally', True) # Shuffle the aggregated data
        )
        
        num_train_batches = len(train_loader) if train_loader else 0
        num_val_batches = len(val_loader) if val_loader else 0
        logger.info(f"[DAggerTrainer] Created dataloaders with {num_train_batches} train, {num_val_batches} val batches.")
        return train_loader, val_loader


    def train_dagger_iteration(self, iteration_num): # Renamed parameter
        logger.info(f"[DAggerTrainer] Starting training for DAgger iteration {iteration_num}")
        
        current_iter_output_dir = os.path.join(self.output_dir, f'training_iter_{iteration_num}') # --- MODIFIED --- path
        os.makedirs(current_iter_output_dir, exist_ok=True)

        training_config_iter = self.config.copy() # Start with global DAgger config
        training_config_iter.update({
            'output_dir': current_iter_output_dir, # Override output dir for this iteration's BCTrainer
            'epochs': self.config.get('dagger_iteration_epochs', 20), 
            'learning_rate': self._get_learning_rate_for_iteration(iteration_num),
            # Pass other relevant BC training params from DAgger config if they differ per iteration
            'batch_size': self.config.get('batch_size', 32),
            'num_workers': self.config.get('train_num_workers', 4),
            'model_type': self.config.get('model_type', 'full'),
            'image_size': self.config.get('image_size', [60,90]),
            'max_depth': self.config.get('max_depth', 10.0),
            'max_velocity': self.config.get('max_velocity', 2.0)
            # ... etc.
        })
        
        # Save this iteration's specific training config
        with open(os.path.join(current_iter_output_dir, 'bc_train_config_this_iter.json'), 'w') as f:
            json.dump(training_config_iter, f, indent=2)

        bc_trainer_for_iter = BCTrainer(training_config_iter)
        
        # --- MODIFIED --- Override BCTrainer's data loading to use our aggregated DAgger data
        # This lambda ensures the BCTrainer instance uses the DAggerTrainer's method
        bc_trainer_for_iter.load_data = lambda: self._assign_dataloaders_to_trainer(bc_trainer_for_iter)
        
        # Load model from previous DAgger iteration if applicable
        if iteration_num > 0: # For first DAgger iter (iter 1), load from initial BC model.
                              # If iter_num is 0, it implies training on initial_bc_data only.
            # Path to best model from iteration (iteration_num - 1)
            # If iteration_num == 1, this means load from iteration 0 (the initial BC model)
            # If iteration_num == 0, this block is skipped.
            prev_iter_num_for_model = iteration_num -1
            if prev_iter_num_for_model == 0 and 'initial_bc_model_path' in self.config:
                 # Special case for first DAgger iteration, load from a specified initial BC model path
                prev_model_path = self.config['initial_bc_model_path']
            elif prev_iter_num_for_model > 0 : # For DAgger iter 2 onwards
                prev_model_path = os.path.join(self.output_dir, f'training_iter_{prev_iter_num_for_model}', 'best_model.pt')
            else: # Should not happen if iteration_num starts at 1 for DAgger iterations
                prev_model_path = None

            if prev_model_path and os.path.exists(prev_model_path):
                logger.info(f"[DAggerTrainer] Loading model weights from: {prev_model_path} for iter {iteration_num}")
                self._load_model_weights_into_trainer(bc_trainer_for_iter, prev_model_path)
            else:
                logger.warning(f"[DAggerTrainer] Previous model path not found or specified ({prev_model_path}) for iter {iteration_num}. Training from scratch or default init.")
        
        bc_trainer_for_iter.train() # This will call the overridden load_data
        
        iteration_summary_info = {
            'iteration_num': iteration_num,
            'timestamp': datetime.now().isoformat(),
            'training_config_used': training_config_iter,
            'dataset_sources_info': self.aggregated_dataset_sources.copy(), # Snapshot of sources used
            'best_val_loss': bc_trainer_for_iter.best_val_loss,
            'best_model_path': bc_trainer_for_iter.best_model_path
        }
        self.iteration_history.append(iteration_summary_info)
        self._save_overall_dagger_progress()
        
        return bc_trainer_for_iter.best_model_path


    def _get_learning_rate_for_iteration(self, iteration_num):
        base_lr = self.config.get('base_learning_rate', self.config.get('learning_rate', 1e-4)) # Allow a DAgger specific base LR
        strategy = self.config.get('dagger_lr_strategy', 'decay')
        
        if strategy == 'constant':
            return base_lr
        elif strategy == 'decay':
            decay_factor = self.config.get('dagger_lr_decay_factor', 0.85) # More specific key
            # iteration_num for DAgger starts at 1. Iteration 0 could be initial BC training.
            # If iteration_num is the DAgger iter number (1, 2, 3...)
            return base_lr * (decay_factor ** (iteration_num -1 if iteration_num > 0 else 0))
        else:
            return base_lr

    def _assign_dataloaders_to_trainer(self, bc_trainer_instance):
        """Helper to assign aggregated dataloaders to a BCTrainer instance."""
        train_loader, val_loader = self._create_aggregated_dataloader_for_training()
        bc_trainer_instance.train_loader = train_loader
        bc_trainer_instance.val_loader = val_loader
        logger.info(f"[DAggerTrainer] Aggregated dataloaders assigned to BCTrainer for iteration.")


    def _load_model_weights_into_trainer(self, bc_trainer_instance, model_path_to_load):
        try:
            checkpoint = torch.load(model_path_to_load, map_location=bc_trainer_instance.device)
            # Ensure model architecture matches if loading only state_dict
            # BCTrainer creates its own model based on its config.
            bc_trainer_instance.model.load_state_dict(checkpoint['model_state_dict'])
            # Optionally load optimizer state if continuing training similarly
            if self.config.get('dagger_load_optimizer_state', False) and 'optimizer_state_dict' in checkpoint :
                bc_trainer_instance.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.info(f"[DAggerTrainer] Loaded optimizer state from {model_path_to_load}")

            logger.info(f"[DAggerTrainer] Model weights successfully loaded into BCTrainer from {model_path_to_load}")
        except Exception as e:
            logger.error(f"[DAggerTrainer] Failed to load model weights from {model_path_to_load}: {e}", exc_info=True)


    def _save_overall_dagger_progress(self):
        progress_filepath = os.path.join(self.output_dir, 'dagger_training_progress.json')
        progress_data_to_save = {
            'total_dagger_iterations_completed': self.current_dagger_iteration,
            'iteration_history': self.iteration_history,
            'last_updated': datetime.now().isoformat()
        }
        with open(progress_filepath, 'w') as f:
            json.dump(progress_data_to_save, f, indent=2, default=str) # default=str for datetime etc.

    def run_dagger_pipeline(self, num_dagger_iterations):
        logger.info(f"[DAggerTrainer] Starting DAgger pipeline for {num_dagger_iterations} iterations.")
        
        # Step 1: Load initial BC dataset
        try:
            self.load_initial_bc_dataset()
        except FileNotFoundError as e:
            logger.error(f"Cannot start DAgger pipeline: {e}")
            return

        # Step 2: Optionally, train an initial BC model (Iteration 0) if not already done
        # Or, ensure 'initial_bc_model_path' is set in config if skipping initial training here.
        if self.config.get('train_initial_bc_model_in_dagger_pipeline', False):
            logger.info("[DAggerTrainer] Training initial BC model (Iteration 0)...")
            # Note: train_dagger_iteration expects iteration_num >= 1 for loading previous model logic.
            # This needs careful handling or a separate initial training step.
            # For now, let's assume an initial_bc_model_path is provided or iter 0 is special.
            # A simple way:
            if 'initial_bc_model_path' not in self.config or not os.path.exists(self.config['initial_bc_model_path']):
                logger.info("No pre-trained initial BC model path found/specified. Training Iteration 0 model now.")
                initial_model_path = self.train_dagger_iteration(iteration_num=0) # Iter 0 only uses initial_bc_data
                self.config['initial_bc_model_path'] = initial_model_path # Save for next iter
                logger.info(f"Initial BC model (Iteration 0) trained. Model at: {initial_model_path}")
            else:
                logger.info(f"Using pre-existing initial BC model from: {self.config['initial_bc_model_path']}")
        elif 'initial_bc_model_path' not in self.config or not os.path.exists(self.config['initial_bc_model_path']):
            logger.error("DAgger pipeline requires 'initial_bc_model_path' in config if not training initial model here.")
            return


        # Step 3: DAgger Iterations
        for i in range(1, num_dagger_iterations + 1):
            self.current_dagger_iteration = i
            logger.info(f"\n{'-'*20} DAgger Iteration: {i} {'-'*20}")

            # 3a. Collect data using current best model
            # This part is external: run data_collection_node.py
            # It needs the path to the model trained in the *previous* iteration.
            model_for_collection_path = self.get_model_path_for_iteration(i-1) # i-1 gives model from prev iter
            if not model_for_collection_path or not os.path.exists(model_for_collection_path):
                logger.error(f"Model for DAgger data collection (iter {i-1}) not found at {model_for_collection_path}. Cannot proceed.")
                break
            
            logger.info(f"RUN DAggerDataCollector for Iteration {i} using model: {model_for_collection_path}")
            logger.info("Please run data_collection_node.py, then provide the path to the collected .pkl file.")
            
            # --- SIMULATE EXTERNAL STEP ---
            # In a real script, you'd have a mechanism to get this path (e.g., user input, file watching)
            dagger_pkl_file_path = input(f"Enter path to DAgger .pkl file for iteration {i} (or type 'skip'): ").strip()
            if dagger_pkl_file_path.lower() == 'skip':
                logger.info(f"Skipping DAgger iteration {i} data aggregation and retraining.")
                continue
            if not os.path.exists(dagger_pkl_file_path):
                logger.error(f"Provided DAgger .pkl file not found: {dagger_pkl_file_path}. Skipping iteration {i}.")
                continue
            # --- END SIMULATE ---

            # 3b. Add collected data to aggregate dataset
            self.add_dagger_collected_data(dagger_pkl_file_path, iteration=i)

            # 3c. Retrain model on aggregated dataset
            logger.info(f"Retraining model for DAgger iteration {i}...")
            new_best_model_path = self.train_dagger_iteration(iteration_num=i)
            logger.info(f"DAgger Iteration {i} retraining complete. New best model at: {new_best_model_path}")
            
            # 3d. Optional: Cleanup old converted datasets to save space
            if i > self.config.get('dagger_keep_converted_datasets', 3) +1 : # Keep N + current + prev
                 self.cleanup_old_converted_datasets(keep_last_n=self.config.get('dagger_keep_converted_datasets', 3))


        logger.info(f"[DAggerTrainer] DAgger pipeline completed for {self.current_dagger_iteration} iterations.")

    def get_model_path_for_iteration(self, iteration_num_trained):
        """Gets the best model path from a completed training iteration."""
        if iteration_num_trained < 0: return None
        if iteration_num_trained == 0: # Initial BC model
            return self.config.get('initial_bc_model_path') 
        
        # For DAgger iterations > 0
        # Check history first
        for hist_item in reversed(self.iteration_history):
            if hist_item['iteration_num'] == iteration_num_trained:
                return hist_item['best_model_path']
        # Fallback to direct path construction
        expected_path = os.path.join(self.output_dir, f'training_iter_{iteration_num_trained}', 'best_model.pt')
        return expected_path if os.path.exists(expected_path) else None


    def cleanup_old_converted_datasets(self, keep_last_n=3):
        """Clean up old CONVERTED DAgger dataset directories (CSV/PNG)."""
        converted_dirs = sorted([
            d for d in os.listdir(self.output_dir)
            if d.startswith('converted_dagger_dataset_iter_') and os.path.isdir(os.path.join(self.output_dir, d))
        ], key=lambda x: int(x.split('_')[-1])) # Sort by iteration number

        if len(converted_dirs) > keep_last_n:
            dirs_to_remove = converted_dirs[:-keep_last_n]
            for dir_name in dirs_to_remove:
                dir_path = os.path.join(self.output_dir, dir_name)
                try:
                    shutil.rmtree(dir_path)
                    logger.info(f"[DAggerTrainer] Cleaned up old converted dataset: {dir_path}")
                except Exception as e:
                    logger.error(f"[DAggerTrainer] Error cleaning up {dir_path}: {e}")


def main_dagger_trainer(): # Renamed
    # --- MODIFIED --- Example config for DAggerTrainer
    dagger_config = {
        'output_dir': os.path.expanduser('~/catkin_ws/src/apf_depth_nav_node/dagger_training_output'),
        'initial_bc_dataset_dir': os.path.expanduser('~/apf_logs/il_dataset/apf_depth_vel'), # Path to your initial CSV/PNG dataset
        'initial_bc_model_path': os.path.expanduser('~/catkin_ws/src/apf_depth_nav_node/scripts/bc_training/full_lr0.0001_seed42_20250603_220558/final_model.pt'), # Path to model trained on initial_bc_dataset_dir
        'train_initial_bc_model_in_dagger_pipeline': True, # Set to True if you want DAggerTrainer to train iter 0

        # BC Training params used by BCTrainer (can be overridden per iteration if needed)
        'model_type': 'full',
        'image_size': [60, 90],
        'max_depth': 10.0,
        'max_velocity': 2.0, # For clamping labels in BCTrainer
        'batch_size': 32,
        'train_num_workers': 4,
        'val_split': 0.2,
        'seed': 42,
        'base_learning_rate': 1e-4, # Base LR for DAgger iterations
        'dagger_iteration_epochs': 15, # Fewer epochs for DAgger fine-tuning

        # DAgger specific params
        'dagger_dataset_weighting_strategy': 'recent_bias', # 'equal' or 'recent_bias'
        'dagger_weight_decay_factor': 0.9, # For 'recent_bias' strategy
        'dagger_lr_strategy': 'decay', # 'constant' or 'decay' for LR across DAgger iterations
        'dagger_lr_decay_factor': 0.85, # For 'decay' LR strategy
        'dagger_load_optimizer_state': False, # Whether to load optimizer state when loading prev model
        'dagger_keep_converted_datasets': 3 # How many old converted CSV/PNG DAgger datasets to keep
    }
    os.makedirs(dagger_config['output_dir'], exist_ok=True)
    
    # Save the main DAgger config
    with open(os.path.join(dagger_config['output_dir'], 'main_dagger_pipeline_config.json'), 'w') as f:
        json.dump(dagger_config, f, indent=2)

    trainer = DAggerTrainer(dagger_config)
    
    num_dagger_iterations_to_run = 1 # Example
    trainer.run_dagger_pipeline(num_dagger_iterations_to_run)
    
    logger.info("[DAggerTrainer] DAgger training process finished.")

if __name__ == '__main__':
    main_dagger_trainer()