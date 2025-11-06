import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DeepQNetwork(nn.Module):
    """DQN with CNN for VNE problem - follows DeepViNE paper architecture"""
    
    def __init__(self, input_shape: tuple, num_actions: int = 9):
        super(DeepQNetwork, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        # CNN layers for feature extraction (as in paper)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # Calculate CNN output size
        conv_out_size = self._get_conv_output(input_shape)
        
        # Dueling DQN architecture (as mentioned in paper)
        self.value_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
    
    def _get_conv_output(self, shape):
        """Calculate CNN output size"""
        with torch.no_grad():
            input = torch.zeros(1, 3, *shape[:2])
            output = self.conv1(input)
            output = F.relu(output)
            output = self.conv2(output)
            output = F.relu(output)
            output = self.conv3(output)
            output = F.relu(output)
            output = self.conv4(output)
            output = F.relu(output)
            return output.numel()  # Use numel() instead of view
    
    def forward(self, x):
        # Convert from (H, W, C) to (C, H, W) for PyTorch
        if len(x.shape) == 3:  # Single image
            x = x.permute(2, 0, 1).unsqueeze(0)
        elif len(x.shape) == 4:  # Batch of images
            x = x.permute(0, 3, 1, 2)
        
        # CNN feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        # Flatten - use reshape instead of view for safety
        x = x.reshape(x.size(0), -1)
        
        # Dueling architecture
        value = self.value_stream(x)
        advantages = self.advantage_stream(x)
        
        # Combine value and advantages
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values

def test_dqn():
    """Test the DQN model"""
    input_shape = (8, 8, 3)  # From our state encoder
    num_actions = 9
    model = DeepQNetwork(input_shape, num_actions=num_actions)
    
    # Create dummy input - ensure it's float32
    dummy_state = torch.randn(1, 8, 8, 3, dtype=torch.float32)
    q_values = model(dummy_state)
    
    print(f"DQN Test:")
    print(f"Input shape: {dummy_state.shape}")
    print(f"Output Q-values shape: {q_values.shape}")
    print(f"Number of actions: {q_values.shape[1]}")
    print(f"Q-values sample: {q_values.detach().numpy()[0][:3]}...")
    
    # Print model summary
    print(f"\nModel Architecture:")
    print(f"CNN layers: 4 conv layers")
    print(f"Input shape: {input_shape}")
    print(f"Output actions: {num_actions}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    return model

if __name__ == "__main__":
    test_dqn()
