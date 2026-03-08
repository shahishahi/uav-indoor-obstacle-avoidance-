import torch
import torch.nn as nn
import torch.nn.functional as F

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class FullBCModel(nn.Module):
    def __init__(self, image_channels=1, velocity_dim=4, output_dim=3, dropout_rate=0.2):
        super().__init__()
        self.image_channels = image_channels
        self.velocity_dim = velocity_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.cnn_encoded_dim = 6144 # Based on CNN architecture below

        # CNN Backbone for image processing
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=5, stride=2, padding=2) # (B, 64, 30, 45)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # (B, 64, 15, 22)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1) # (B, 128, 15, 22)
        self.pool2 = nn.MaxPool2d(kernel_size=(3,2), stride=(3,2)) # (B, 128, 5, 11)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(2,6), stride=1, padding=0) # (B, 256, 4, 6)
        # Flattened size: 256 * 4 * 6 = 6144, matches self.cnn_encoded_dim

        # Verify CNN output dimension
        with torch.no_grad():
            dummy_img = torch.zeros(1, image_channels, 60, 90)
            x = F.relu(self.conv1(dummy_img))
            x = self.pool1(x)
            x = F.relu(self.conv2(x))
            x = self.pool2(x)
            x = F.relu(self.conv3(x))
            cnn_flat_dim_actual = x.flatten(1).shape[1]
            if cnn_flat_dim_actual != self.cnn_encoded_dim:
                print(f"[FullBCModel WARNING] Calculated CNN flattened dim {cnn_flat_dim_actual} does not match target cnn_encoded_dim {self.cnn_encoded_dim}. Check architecture.")
                self.cnn_encoded_dim = cnn_flat_dim_actual # Use actual if different

        # MLP Head
        mlp_input_dim = self.cnn_encoded_dim + velocity_dim
        hidden_dim_mlp = 253 # Calculated to match total parameters in original spec
        
        self.fc1 = nn.Linear(mlp_input_dim, hidden_dim_mlp)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim_mlp, output_dim)

    def forward(self, depth_img, velocity):
        # Image processing
        x_img = F.relu(self.conv1(depth_img))
        x_img = self.pool1(x_img)
        x_img = F.relu(self.conv2(x_img))
        x_img = self.pool2(x_img)
        x_img = F.relu(self.conv3(x_img))
        
        encoded_img = x_img.flatten(1) # Flatten all dimensions except batch

        # Concatenate image features and velocity
        combined_features = torch.cat((encoded_img, velocity), dim=1)
        
        # MLP
        x = F.relu(self.fc1(combined_features))
        x = self.dropout(x)
        output = self.fc2(x)
        
        return output

    def get_model_info(self):
        return {
            "model_class": self.__class__.__name__,
            "image_channels": self.image_channels,
            "velocity_dim": self.velocity_dim,
            "output_dim": self.output_dim,
            "cnn_encoded_dim": self.cnn_encoded_dim,
            "dropout_rate": self.dropout_rate,
            "total_trainable_params": count_parameters(self)
        }

class LightweightBCModel(nn.Module):
    def __init__(self, image_channels=1, velocity_dim=4, output_dim=3, dropout_rate=0.2):
        super().__init__()
        self.image_channels = image_channels
        self.velocity_dim = velocity_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        
        # Lightweight CNN Backbone
        self.conv1_lw = nn.Conv2d(image_channels, 16, kernel_size=5, stride=2, padding=2) # (B, 16, 30, 45)
        self.pool1_lw = nn.MaxPool2d(kernel_size=2, stride=2) # (B, 16, 15, 22)
        
        self.conv2_lw = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1) # (B, 32, 15, 22)
        self.pool2_lw = nn.MaxPool2d(kernel_size=2, stride=2) # (B, 32, 7, 11)
        
        self.conv3_lw = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # (B, 64, 7, 11)
        self.pool3_lw = nn.MaxPool2d(kernel_size=2, stride=2) # (B, 64, 3, 5)
        
        # Flattened size: 64 * 3 * 5 = 960
        self.cnn_encoded_dim = 64 * 3 * 5
        
        # MLP Head
        mlp_input_dim_lw = self.cnn_encoded_dim + velocity_dim
        hidden_dim_mlp_lw = 409 # Calculated to match target parameters in original spec
        
        self.fc1_lw = nn.Linear(mlp_input_dim_lw, hidden_dim_mlp_lw)
        self.dropout_lw = nn.Dropout(dropout_rate)
        self.fc2_lw = nn.Linear(hidden_dim_mlp_lw, output_dim)

    def forward(self, depth_img, velocity):
        # Image processing
        x_img = F.relu(self.conv1_lw(depth_img))
        x_img = self.pool1_lw(x_img)
        x_img = F.relu(self.conv2_lw(x_img))
        x_img = self.pool2_lw(x_img)
        x_img = F.relu(self.conv3_lw(x_img))
        x_img = self.pool3_lw(x_img)
        
        encoded_img = x_img.flatten(1)

        # Concatenate image features and velocity
        combined_features = torch.cat((encoded_img, velocity), dim=1)
        
        # MLP
        x = F.relu(self.fc1_lw(combined_features))
        x = self.dropout_lw(x)
        output = self.fc2_lw(x)
        
        return output

    def get_model_info(self):
        return {
            "model_class": self.__class__.__name__,
            "image_channels": self.image_channels,
            "velocity_dim": self.velocity_dim,
            "output_dim": self.output_dim,
            "cnn_encoded_dim": self.cnn_encoded_dim,
            "dropout_rate": self.dropout_rate,
            "total_trainable_params": count_parameters(self)
        }

def create_model(model_type='full', image_channels=1, velocity_dim=4, output_dim=3, dropout_rate=0.2, **kwargs):
    """
    Factory function to create a BC model.
    **kwargs can be used for additional model-specific parameters if needed in the future.
    """
    if model_type == 'full':
        model = FullBCModel(image_channels=image_channels, velocity_dim=velocity_dim, output_dim=output_dim, dropout_rate=dropout_rate)
    elif model_type == 'lightweight':
        model = LightweightBCModel(image_channels=image_channels, velocity_dim=velocity_dim, output_dim=output_dim, dropout_rate=dropout_rate)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'full' or 'lightweight'.")
    return model

if __name__ == "__main__":
    BATCH_SIZE = 4
    IMG_C, IMG_H, IMG_W = 1, 60, 90
    VEL_DIM = 4
    OUTPUT_DIM = 3
    DROPOUT_RATE = 0.25 # Example different dropout for testing

    print("--- Full Model Test ---")
    full_model = create_model(
        model_type='full', 
        image_channels=IMG_C, 
        velocity_dim=VEL_DIM, 
        output_dim=OUTPUT_DIM, 
        dropout_rate=DROPOUT_RATE
    )
    model_info_full = full_model.get_model_info()
    print("Full Model Info from get_model_info():")
    for key, value in model_info_full.items():
        print(f"  {key}: {value}")
    
    # Test forward pass
    dummy_depth_full = torch.randn(BATCH_SIZE, IMG_C, IMG_H, IMG_W)
    dummy_velocity_full = torch.randn(BATCH_SIZE, VEL_DIM)
    output_full = full_model(dummy_depth_full, dummy_velocity_full)
    print(f"  Input depth shape: {dummy_depth_full.shape}")
    print(f"  Input velocity shape: {dummy_velocity_full.shape}")
    print(f"  Output shape: {output_full.shape}")
    print("\n==================================================")

    print("--- Lightweight Model Test ---")
    lightweight_model = create_model(
        model_type='lightweight',
        image_channels=IMG_C,
        velocity_dim=VEL_DIM,
        output_dim=OUTPUT_DIM,
        dropout_rate=DROPOUT_RATE
    )
    model_info_lw = lightweight_model.get_model_info()
    print("Lightweight Model Info from get_model_info():")
    for key, value in model_info_lw.items():
        print(f"  {key}: {value}")

    total_params_full = model_info_full["total_trainable_params"]
    total_params_lw = model_info_lw["total_trainable_params"]
    if total_params_full > 0:
        size_reduction = (1 - total_params_lw / total_params_full) * 100
        print(f"  Size reduction compared to Full Model: {size_reduction:.1f}%")
    else:
        print("  Full model has 0 parameters, cannot calculate size reduction.")


    # Test forward pass
    dummy_depth_lw = torch.randn(BATCH_SIZE, IMG_C, IMG_H, IMG_W)
    dummy_velocity_lw = torch.randn(BATCH_SIZE, VEL_DIM)
    output_lw = lightweight_model(dummy_depth_lw, dummy_velocity_lw)
    print(f"  Input depth shape: {dummy_depth_lw.shape}")
    print(f"  Input velocity shape: {dummy_velocity_lw.shape}")
    print(f"  Output shape: {output_lw.shape}")