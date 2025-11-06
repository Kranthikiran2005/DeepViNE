import numpy as np
import random
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add src to path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from environment.network import PhysicalNetwork, VirtualNetwork
from models.state_encoder import StateEncoder

class VNEEnvironment:
    """Virtual Network Embedding Environment with DeepViNE action system"""
    
    def __init__(self, grid_size: Tuple[int, int] = (5, 5), difficulty: str = "hard"):
        self.grid_size = grid_size
        self.rows, self.cols = grid_size
        self.difficulty = difficulty  # "easy", "hard", or "mixed"
        self.physical_network = None
        self.current_vn = None
        self.embedded_vns: List[VirtualNetwork] = []
        self.vn_counter = 0
        
        # DeepViNE pointers and state
        self.virtual_pointer = 0
        self.physical_pointer = 0
        self.embedded_nodes: List[int] = []  # Track embedded virtual nodes
        self.current_iteration = 0
        self.max_iterations = 50  # Prevent infinite loops
        
        # State encoder
        self.state_encoder = StateEncoder(physical_grid_size=grid_size, 
                                        virtual_grid_size=(3, 3))
        
        # Metrics
        self.total_revenue = 0
        self.total_cost = 0
        self.blocked_vns = 0
        self.total_vns = 0
        
        self.reset()
    
    def reset(self):
        """Reset the environment for a new episode"""
        self.physical_network = self._create_grid_network()
        self.embedded_vns = []
        self.vn_counter = 0
        self.current_vn = None
        
        # Reset pointers and state
        self.virtual_pointer = 0
        self.physical_pointer = 0
        self.embedded_nodes = []
        self.current_iteration = 0
        
        # Reset metrics
        self.total_revenue = 0
        self.total_cost = 0
        self.blocked_vns = 0
        self.total_vns = 0
        
        return self._get_state()
    
    def _create_grid_network(self) -> PhysicalNetwork:
    	"""Create physical network with balanced capacity"""
    	pn = PhysicalNetwork()
    	pn.grid_size = self.grid_size
    
    	# Create nodes with BALANCED capacities
    	for i in range(self.rows * self.cols):
            if self.difficulty == "hard":
            	cpu_capacity = random.uniform(80, 150)  # Limited for hard problems
            else:
            	cpu_capacity = random.uniform(100, 200)  # Standard capacity
            pn.add_node(i, cpu_capacity)
    
    	# Create grid links with BALANCED capacities
    	for i in range(self.rows):
            for j in range(self.cols):
            	node_id = i * self.cols + j
            
            	# Right neighbor
            	if j < self.cols - 1:
                    right_id = i * self.cols + (j + 1)
                    if self.difficulty == "hard":
                    	bw_capacity = random.uniform(60, 100)
                    else:
                    	bw_capacity = random.uniform(50, 100)
                    pn.add_link(node_id, right_id, bw_capacity)
            
            	# Bottom neighbor
            	if i < self.rows - 1:
                    bottom_id = (i + 1) * self.cols + j
                    if self.difficulty == "hard":
                    	bw_capacity = random.uniform(60, 100)
                    else:
                    	bw_capacity = random.uniform(50, 100)
                    pn.add_link(node_id, bottom_id, bw_capacity)
    
    	return pn
    
    def generate_virtual_network(self) -> VirtualNetwork:
    	"""Generate moderate virtual network requests"""
    	vn = VirtualNetwork(self.vn_counter, 0, 100)
    	self.vn_counter += 1
    
    	# Create 3x3 grid with MODERATE demands
    	grid_nodes = 9
    	for i in range(grid_nodes):
            cpu_demand = random.uniform(1, 5)  # Moderate: 5-20 CPU
            vn.add_node(i, cpu_demand)
    
    	# Create grid links with MODERATE demands
    	for i in range(3):
            for j in range(3):
            	node_id = i * 3 + j
            
            	# Right neighbor
            	if j < 2:
                    right_id = i * 3 + (j + 1)
                    bw_demand = random.uniform(1, 5)
                    vn.add_link(node_id, right_id, bw_demand)
            
            	# Bottom neighbor
            	if i < 2:
                    bottom_id = (i + 1) * 3 + j
                    bw_demand = random.uniform(1, 5)
                    vn.add_link(node_id, bottom_id, bw_demand)
    
    	return vn

    def generate_difficult_virtual_network(self) -> VirtualNetwork:
    	"""Generate challenging but solvable virtual networks"""
    	vn = VirtualNetwork(self.vn_counter, 0, 100)
    	self.vn_counter += 1
    
    	# Create 3x3 grid with CHALLENGING demands
    	grid_nodes = 9
    	for i in range(grid_nodes):
            cpu_demand = random.uniform(10, 30)  # Challenging: 15-40 CPU
            vn.add_node(i, cpu_demand)
    
    	# Create grid links with CHALLENGING demands
    	for i in range(3):
            for j in range(3):
            	node_id = i * 3 + j
            
            	# Right neighbor
            	if j < 2:
                    right_id = i * 3 + (j + 1)
                    bw_demand = random.uniform(10, 30)
                    vn.add_link(node_id, right_id, bw_demand)
            
            	# Bottom neighbor
            	if i < 2:
                    bottom_id = (i + 1) * 3 + j
                    bw_demand = random.uniform(10, 30)
                    vn.add_link(node_id, bottom_id, bw_demand)
    
    	return vn
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one environment step based on DeepViNE actions
        
        Actions 0-3: Move virtual pointer (up, right, down, left)
        Actions 4-7: Move physical pointer (up, right, down, left) 
        Action 8: Embed current virtual node at current physical node
        """
        reward = 0
        done = False
        info = {}
        
        self.current_iteration += 1
        
        # Action processing based on DeepViNE paper
        if action < 4:
            # Move virtual pointer (0-3: up, right, down, left)
            self._move_virtual_pointer(action)
            reward = 0.0  # No reward for moving pointers
            
        elif action < 8:
            # Move physical pointer (4-7: up, right, down, left)
            self._move_physical_pointer(action - 4)
            reward = 0.0  # No reward for moving pointers
            
        elif action == 8:
            # Embed action
            reward = self._try_embed()
            if reward > 0:
                # Check if all virtual nodes are embedded
                if len(self.embedded_nodes) == len(self.current_vn.nodes):
                    reward = 1.0  # Full VN embedded successfully
                    done = True
                else:
                    # Move to next virtual node
                    self._move_to_next_virtual_node()
            else:
                # Embedding failed
                self.blocked_vns += 1
                done = True
        
        # Check for timeout
        if self.current_iteration >= self.max_iterations:
            reward = -0.5  # Penalty for timeout
            done = True
            self.blocked_vns += 1
        
        next_state = self._get_state()
        info = self.get_metrics()
        
        return next_state, reward, done, info
    
    def _move_virtual_pointer(self, direction: int):
        """Move virtual pointer in grid"""
        current_row = self.virtual_pointer // 3
        current_col = self.virtual_pointer % 3
        
        # Directions: 0=up, 1=right, 2=down, 3=left
        if direction == 0 and current_row > 0:  # Up
            self.virtual_pointer -= 3
        elif direction == 1 and current_col < 2:  # Right
            self.virtual_pointer += 1
        elif direction == 2 and current_row < 2:  # Down
            self.virtual_pointer += 3
        elif direction == 3 and current_col > 0:  # Left
            self.virtual_pointer -= 1
    
    def _move_physical_pointer(self, direction: int):
        """Move physical pointer in grid"""
        current_row = self.physical_pointer // self.cols
        current_col = self.physical_pointer % self.cols
        
        # Directions: 0=up, 1=right, 2=down, 3=left
        if direction == 0 and current_row > 0:  # Up
            self.physical_pointer -= self.cols
        elif direction == 1 and current_col < self.cols - 1:  # Right
            self.physical_pointer += 1
        elif direction == 2 and current_row < self.rows - 1:  # Down
            self.physical_pointer += self.cols
        elif direction == 3 and current_col > 0:  # Left
            self.physical_pointer -= 1
    
    def _move_to_next_virtual_node(self):
        """Move to next unembedded virtual node"""
        for i in range(len(self.current_vn.nodes)):
            if i not in self.embedded_nodes:
                self.virtual_pointer = i
                break
    
    def _try_embed(self) -> float:
        """Try to embed current virtual node at current physical node"""
        if self.current_vn is None:
            return -0.1  # No VN to embed
        
        v_node_id = self.virtual_pointer
        p_node_id = self.physical_pointer
        
        if v_node_id in self.embedded_nodes:
            return -0.1  # Already embedded
        
        v_node = self.current_vn.nodes[v_node_id]
        p_node = self.physical_network.nodes[p_node_id]
        
        # Check if physical node has sufficient capacity
        if p_node.available_cpu >= v_node.cpu_capacity:
            # Embed the virtual node
            p_node.allocate_cpu(v_node.cpu_capacity)
            self.embedded_nodes.append(v_node_id)
            
            # Update revenue and cost (simplified)
            self.total_revenue += v_node.cpu_capacity
            self.total_cost += v_node.cpu_capacity
            
            return 0.1  # Small positive reward for successful embedding
        else:
            return -0.1  # Negative reward for failed embedding
    
    def _get_state(self) -> np.ndarray:
        """Get current state as image representation"""
        if self.current_vn is None:
            # Choose network difficulty based on setting
            if self.difficulty == "easy":
                self.current_vn = self.generate_virtual_network()
            elif self.difficulty == "hard":
                self.current_vn = self.generate_difficult_virtual_network()
            else:  # mixed
                # Use difficult networks 50% of the time for mixed difficulty
                if random.random() < 0.5:
                    self.current_vn = self.generate_difficult_virtual_network()
                else:
                    self.current_vn = self.generate_virtual_network()
            
            self.total_vns += 1
        
        return self.state_encoder.encode_state(
            physical_net=self.physical_network,
            virtual_net=self.current_vn,
            virtual_pointer=self.virtual_pointer,
            physical_pointer=self.physical_pointer,
            embedded_nodes=self.embedded_nodes
        )
    
    def get_metrics(self) -> Dict:
        """Get performance metrics"""
        acceptance_rate = (self.total_vns - self.blocked_vns) / self.total_vns if self.total_vns > 0 else 0
        
        return {
            'total_vns': self.total_vns,
            'blocked_vns': self.blocked_vns,
            'acceptance_rate': acceptance_rate,
            'total_revenue': self.total_revenue,
            'total_cost': self.total_cost,
            'current_iteration': self.current_iteration,
            'embedded_nodes': len(self.embedded_nodes)
        }

def test_environment():
    """Test the complete VNE environment with different difficulties"""
    print("Testing VNE Environment with Different Difficulties:")
    
    # Test easy environment
    print("\n1. Easy Environment:")
    env_easy = VNEEnvironment(grid_size=(5, 5), difficulty="easy")
    state = env_easy.reset()
    vn = env_easy.current_vn
    print(f"   VN CPU Demand: {sum(node.cpu_capacity for node in vn.nodes.values()):.1f}")
    
    # Test hard environment  
    print("\n2. Hard Environment:")
    env_hard = VNEEnvironment(grid_size=(5, 5), difficulty="mixed")
    state = env_hard.reset()
    vn = env_hard.current_vn
    print(f"   VN CPU Demand: {sum(node.cpu_capacity for node in vn.nodes.values()):.1f}")
    
    # Test mixed environment
    print("\n3. Mixed Environment:")
    env_mixed = VNEEnvironment(grid_size=(5, 5), difficulty="hard")
    
    # Test a few actions
    actions = [1, 5, 8]  # Move virtual right, move physical right, embed
    for i, action in enumerate(actions):
        next_state, reward, done, info = env_mixed.step(action)
        print(f"   Action {action}: reward={reward:.3f}, done={done}, embedded_nodes={info['embedded_nodes']}")
        
        if done:
            break
    
    print(f"   Final metrics: {env_mixed.get_metrics()}")
    return env_mixed

if __name__ == "__main__":
    test_environment()