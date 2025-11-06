import numpy as np
from typing import Dict, List, Tuple
import sys
import os

# Add src to path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class StateEncoder:
    """Encodes the VNE problem state as a 3-channel image following DeepViNE paper"""
    
    def __init__(self, physical_grid_size: Tuple[int, int] = (5, 5), 
                 virtual_grid_size: Tuple[int, int] = (3, 3)):
        self.physical_grid_size = physical_grid_size
        self.virtual_grid_size = virtual_grid_size
        self.physical_rows, self.physical_cols = physical_grid_size
        self.virtual_rows, self.virtual_cols = virtual_grid_size
        
    def encode_state(self, 
                    physical_net,
                    virtual_net,
                    virtual_pointer: int,
                    physical_pointer: int,
                    embedded_nodes: List[int]) -> np.ndarray:
        """
        Encode the current state as a 3-channel image per DeepViNE paper
        
        Channel 1: Basic resource information and node status
        Channel 2: Embedding completion status  
        Channel 3: Sufficient capacity indicators
        """
        # Calculate dimensions - concatenate physical and virtual networks
        total_rows = self.physical_rows + self.virtual_rows
        total_cols = self.physical_cols + self.virtual_cols
        
        # Initialize the 3 channels
        channel1 = np.zeros((total_rows, total_cols))
        channel2 = np.zeros((total_rows, total_cols))
        channel3 = np.zeros((total_rows, total_cols))
        
        # Encode physical network in upper section
        self._encode_physical_network(physical_net, physical_pointer, 
                                    channel1, channel2, channel3, 
                                    row_offset=0, col_offset=0)
        
        # Encode virtual network in lower section  
        self._encode_virtual_network(virtual_net, virtual_pointer, embedded_nodes,
                                   channel1, channel2, channel3,
                                   row_offset=self.physical_rows, col_offset=0)
        
        # Stack channels to create 3-channel image
        state_image = np.stack([channel1, channel2, channel3], axis=-1)
        return state_image
    
    def _encode_physical_network(self, physical_net, physical_pointer: int,
                               channel1: np.ndarray, channel2: np.ndarray, channel3: np.ndarray,
                               row_offset: int, col_offset: int):
        """Encode physical network according to paper's Channel 1 specification"""
        rows, cols = self.physical_grid_size
        
        for i in range(rows):
            for j in range(cols):
                node_id = i * cols + j
                if node_id in physical_net.nodes:
                    node = physical_net.nodes[node_id]
                    abs_row = row_offset + i
                    abs_col = col_offset + j
                    
                    # Channel 1: Available CPU (normalized)
                    if node.available_cpu > 0:
                        channel1[abs_row, abs_col] = node.available_cpu / 100.0  # Normalize
                    
                    # Channel 1: Pointer indicator
                    if node_id == physical_pointer:
                        channel1[abs_row, abs_col] = 1.0  # Max value for pointer
                    
                    # Channel 3: Sufficient capacity indicator
                    channel3[abs_row, abs_col] = 1.0 if node.available_cpu > 5 else 0.0
    
    def _encode_virtual_network(self, virtual_net, virtual_pointer: int, 
                              embedded_nodes: List[int],
                              channel1: np.ndarray, channel2: np.ndarray, channel3: np.ndarray,
                              row_offset: int, col_offset: int):
        """Encode virtual network according to paper's specification"""
        rows, cols = self.virtual_grid_size
        
        for i in range(rows):
            for j in range(cols):
                node_id = i * cols + j
                if node_id in virtual_net.nodes:
                    node = virtual_net.nodes[node_id]
                    abs_row = row_offset + i
                    abs_col = col_offset + j
                    
                    # Channel 1: CPU demand (normalized)
                    channel1[abs_row, abs_col] = node.cpu_capacity / 10.0  # Normalize
                    
                    # Channel 1: Pointer indicator
                    if node_id == virtual_pointer:
                        channel1[abs_row, abs_col] = 1.0  # Max value for pointer
                    
                    # Channel 2: Embedding status
                    channel2[abs_row, abs_col] = 0.0 if node_id in embedded_nodes else 1.0

                    # Channel 3: Always 1 for virtual nodes (they need to be embedded)
                    channel3[abs_row, abs_col] = 1.0