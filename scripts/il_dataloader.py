#!/usr/bin/env python3

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import csv
from os.path import join as opj
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# +++ ADDED +++
# Centralized function for preprocessing depth images specifically for model input
def preprocess_depth_for_model_input(depth_image_meters, target_size=(60, 90), max_depth_value=10.0):
    """
    Preprocesses a depth image (already in meters) for model input.
    - Normalizes by max_depth_value to [0,1]
    - Resizes to target_size
    Args:
        depth_image_meters (np.ndarray): Depth image in float32, units of meters.
        target_size (tuple): (height, width) for resizing.
        max_depth_value (float): Maximum depth value for clipping and normalization.
    Returns:
        np.ndarray: Processed depth image, float32, normalized and resized.
    """
    # 1. Clip to reasonable depth range (already done if input is from self.current_depth in DAgger)
    #    and normalize to [0,1]
    #    Input `depth_image_meters` is assumed to be already somewhat processed (e.g., NaNs handled, clipped if needed from raw sensor)
    #    The main step here is normalization against max_depth_value.
    img_normalized = np.clip(depth_image_meters, 0, max_depth_value) / max_depth_value
    
    # 2. Resize to target size
    # cv2.resize uses (width, height)
    img_resized = cv2.resize(img_normalized, (target_size[1], target_size[0]))
    
    return img_resized


class DepthVelDataset(Dataset):
    """
    Loads depth images, drone's current velocity, and expert velocity labels.
    Consistent preprocessing with training and inference.
    """
    def __init__(self, meta_data, img_dir_map, crop_size=(60, 90), transform=None, max_depth=10.0): # --- MODIFIED --- img_dir to img_dir_map
        self.meta_data = meta_data
        # --- MODIFIED --- Allow mapping for multiple image directories if meta_data comes from multiple sources
        self.img_dir_map = img_dir_map # e.g., {'initial_bc': '/path/to/bc_images', 'dagger_iter1': '/path/to/iter1_images'}
                                     # or just a single string if only one source.
        self.crop_size = crop_size
        self.transform = transform
        self.max_depth = max_depth  # Maximum depth in meters
        
        logger.info(f"Dataset initialized with {len(meta_data)} samples")
        logger.info(f"Image directory map: {img_dir_map}")
        logger.info(f"Crop size: {crop_size}")
        logger.info(f"Max depth for normalization: {max_depth}m")

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        row = self.meta_data[idx]
        
        # --- MODIFIED --- Determine correct image directory based on metadata
        img_source_key = row.get('source_dataset_key', None) # Expect this key if using multiple datasets
        if isinstance(self.img_dir_map, dict) and img_source_key:
            current_img_dir = self.img_dir_map.get(img_source_key)
            if not current_img_dir:
                logger.error(f"Image source key '{img_source_key}' not found in img_dir_map for sample {idx}")
                # Fallback or raise error
                current_img_dir = list(self.img_dir_map.values())[0] if self.img_dir_map else '.'
        elif isinstance(self.img_dir_map, str):
            current_img_dir = self.img_dir_map
        else:
            logger.error(f"Invalid img_dir_map configuration for sample {idx}")
            current_img_dir = '.' # Fallback

        img_path = opj(current_img_dir, row['image_file'])

        # Load depth image (expected to be uint16, mm)
        img_uint16 = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img_uint16 is None:
            logger.error(f"Image not found or failed to load: {img_path}")
            # Return a dummy image and label or raise error to catch issues early
            dummy_img = torch.zeros(1, self.crop_size[0], self.crop_size[1], dtype=torch.float32)
            dummy_vel = torch.zeros(4, dtype=torch.float32)
            dummy_label = torch.zeros(3, dtype=torch.float32)
            return dummy_img, dummy_vel, dummy_label
            # raise FileNotFoundError(f"Image not found: {img_path}")


        # Convert from uint16 (mm) to float32 (meters)
        img_meters = img_uint16.astype(np.float32) / 1000.0
            
        # --- MODIFIED --- Use the centralized preprocessing function for model input preparation
        # This function handles normalization by max_depth and resizing
        img_processed_np = preprocess_depth_for_model_input(
            img_meters,
            target_size=self.crop_size,
            max_depth_value=self.max_depth
        )
        
        # Convert to tensor [1, H, W]
        img_tensor = torch.from_numpy(img_processed_np).unsqueeze(0).float()

        # Drone's current velocity as input (4D: vx, vy, vz, yaw_rate)
        vel_input = torch.tensor([
            float(row['vx_input']),
            float(row['vy_input']),
            float(row['vz_input']),
            float(row['yaw_rate']),
        ], dtype=torch.float32)

        # Expert command (APF output) as label (3D: vx, vy, vz)
        label = torch.tensor([
            float(row['vx']),
            float(row['vy']),
            float(row['vz']),
        ], dtype=torch.float32)

        # Apply transforms if provided (usually not needed for depth if already tensor)
        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor, vel_input, label


def load_dataset_from_sources(dataset_sources, crop_size=(60, 90), val_split=0.2, seed=None, 
                       batch_size=32, num_workers=2, max_depth=10.0, shuffle_train=True):
    """
    Loads training and validation datasets from one or more sources and returns PyTorch DataLoaders.
    Each source is a dictionary like: {'key': 'unique_key', 'csv_path': '/path/to/controls.csv', 'img_dir': '/path/to/images'}
    
    Args:
        dataset_sources (list): List of dataset source dictionaries.
        crop_size: Target image size (height, width)
        val_split: Fraction of data to use for validation
        seed: Random seed for reproducibility
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        max_depth: Maximum depth value in meters for normalization
        shuffle_train: Whether to shuffle training data
    
    Returns:
        train_loader, val_loader: PyTorch DataLoaders
    """
    all_meta_data = []
    img_dir_map = {}

    for source in dataset_sources:
        source_key = source['key']
        csv_path = source['csv_path']
        img_dir = source['img_dir']
        
        img_dir_map[source_key] = img_dir

        if not os.path.exists(csv_path):
            logger.warning(f"Controls CSV not found for source '{source_key}': {csv_path}. Skipping.")
            continue
        if not os.path.exists(img_dir):
            logger.warning(f"Images directory not found for source '{source_key}': {img_dir}. Skipping.")
            continue
            
        logger.info(f"Loading data from source '{source_key}': {csv_path}")
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            source_meta_data = []
            for row in reader:
                row['source_dataset_key'] = source_key # Add key to identify image source
                source_meta_data.append(row)
            
            if not source_meta_data:
                logger.warning(f"No data found in CSV for source '{source_key}': {csv_path}")
                continue

            # Validate CSV format for this source
            required_columns = ['timestamp', 'vx_input', 'vy_input', 'vz_input', 'yaw_rate', 'vx', 'vy', 'vz', 'image_file']
            if source_meta_data: # Check only if data was loaded
                missing_columns = [col for col in required_columns if col not in source_meta_data[0].keys()]
                if missing_columns:
                    logger.error(f"Missing required columns in CSV for source '{source_key}': {missing_columns}. Skipping this source.")
                    continue
            
            all_meta_data.extend(source_meta_data)
            logger.info(f"Loaded {len(source_meta_data)} entries from source '{source_key}'")

    if not all_meta_data:
        raise ValueError("No data loaded from any source. Check paths and CSV formats.")
    
    logger.info(f"Total loaded {len(all_meta_data)} data entries from all sources.")

    # Sort by timestamp for temporal consistency (optional, but can be good)
    all_meta_data.sort(key=lambda x: float(x['timestamp']))
    
    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
        
    # Shuffle data while maintaining some temporal structure or apply weighting
    if shuffle_train:
        # Basic shuffle of all aggregated data.
        # For weighted sampling (e.g. if DAgger iterations have different weights),
        # you'd need a more complex sampler for the DataLoader or oversample/undersample `all_meta_data` here.
        np.random.shuffle(all_meta_data)

    # Split into train and validation
    split_idx = int(len(all_meta_data) * (1 - val_split))
    train_meta = all_meta_data[:split_idx]
    val_meta = all_meta_data[split_idx:]

    logger.info(f"Total train samples: {len(train_meta)}")
    logger.info(f"Total validation samples: {len(val_meta)}")

    # Create Datasets
    train_dataset = DepthVelDataset(train_meta, img_dir_map, crop_size, max_depth=max_depth)
    val_dataset = DepthVelDataset(val_meta, img_dir_map, crop_size, max_depth=max_depth)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle_train, # Shuffling is now done on all_meta_data if shuffle_train is True
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True 
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    logger.info(f"Created DataLoaders with batch_size={batch_size}, num_workers={num_workers}")
    return train_loader, val_loader


# --- MODIFIED --- Kept original load_dataset for single source, useful for initial BC training
def load_dataset(dataset_dir, crop_size=(60, 90), val_split=0.2, seed=None, 
                batch_size=32, num_workers=2, max_depth=10.0, shuffle_train=True):
    """
    Loads training and validation datasets from a single dataset directory.
    """
    img_dir = opj(dataset_dir, 'images')
    csv_path = opj(dataset_dir, 'controls.csv')
    
    single_source = [{
        'key': 'dataset', # A default key
        'csv_path': csv_path,
        'img_dir': img_dir
    }]
    
    return load_dataset_from_sources(
        single_source, crop_size, val_split, seed, 
        batch_size, num_workers, max_depth, shuffle_train
    )


def validate_dataset(dataset_dir, num_samples=5):
    """
    Validate dataset by loading and checking a few samples.
    """
    logger.info(f"Validating dataset: {dataset_dir}")
    
    try:
        train_loader, _ = load_dataset( # Use original load_dataset for simple validation
            dataset_dir=dataset_dir,
            batch_size=min(num_samples, 4),
            num_workers=0 
        )
        
        for i, (images, input_vels, expert_vels) in enumerate(train_loader):
            logger.info(f"Batch {i+1}:")
            logger.info(f"  Image shape: {images.shape}, dtype: {images.dtype}") # Added dtype
            logger.info(f"  Input velocity shape: {input_vels.shape}, dtype: {input_vels.dtype}")
            logger.info(f"  Expert velocity shape: {expert_vels.shape}, dtype: {expert_vels.dtype}")
            logger.info(f"  Image range: [{images.min():.3f}, {images.max():.3f}] (Expected [0,1] after norm)")
            logger.info(f"  Input vel range: [{input_vels.min():.3f}, {input_vels.max():.3f}]")
            logger.info(f"  Expert vel range: [{expert_vels.min():.3f}, {expert_vels.max():.3f}]")
            
            if i >= 1: 
                break
                
        logger.info("Dataset validation completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Dataset validation failed: {e}", exc_info=True) # Added exc_info
        return False


if __name__ == '__main__':
    dataset_dir_main = '/home/shahi/apf_logs/il_dataset/apf_depth_vel' # Example path
    
    if os.path.exists(dataset_dir_main):
        validate_dataset(dataset_dir_main)
        
        # Example of loading from multiple sources (if you had DAgger data in CSV/PNG format)
        # sources_example = [
        #     {'key': 'initial_bc', 'csv_path': opj(dataset_dir_main, 'controls.csv'), 'img_dir': opj(dataset_dir_main, 'images')},
        #     # {'key': 'dagger_iter1', 'csv_path': '/path/to/dagger_iter1/controls.csv', 'img_dir': '/path/to/dagger_iter1/images'}
        # ]
        # try:
        #     train_loader_agg, val_loader_agg = load_dataset_from_sources(sources_example)
        #     logger.info(f"Aggregated train loader has {len(train_loader_agg)} batches.")
        # except ValueError as e:
        #     logger.error(f"Could not load aggregated dataset: {e}")

    else:
        logger.error(f"Dataset directory not found: {dataset_dir_main}")
        logger.info("Please update the dataset path or run bag_to_csv.py first")