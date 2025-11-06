import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.dqn import DeepQNetwork

class DeepViNEAgent:
    """Deep Reinforcement Learning agent for VNE using DQN"""
    
    def __init__(self, state_shape, num_actions=9, learning_rate=0.001):
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Neural networks
        self.policy_net = DeepQNetwork(state_shape, num_actions).to(self.device)
        self.target_net = DeepQNetwork(state_shape, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Experience replay
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        
        # Training parameters
        self.gamma = 0.99  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update = 100  # update target network every 100 steps
        
        self.steps_done = 0
    
    def select_action(self, state):
        """Select action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            # Explore: random action
            return random.randint(0, self.num_actions - 1)
        else:
            # Exploit: use policy network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def train_step(self):
        """Perform one training step"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).unsqueeze(1).to(self.device)
        
        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions)
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Update target network
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def save_model(self, filepath):
        """Save the trained model"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        checkpoint = torch.load(filepath, weights_only=True)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        print(f"Model loaded from {filepath}")

def test_agent():
    """Test the DeepViNE agent"""
    state_shape = (8, 8, 3)
    num_actions = 9
    
    agent = DeepViNEAgent(state_shape, num_actions)
    
    print("DeepViNE Agent Test:")
    print(f"State shape: {state_shape}")
    print(f"Number of actions: {num_actions}")
    print(f"Initial epsilon: {agent.epsilon}")
    print(f"Memory size: {len(agent.memory)}")
    print(f"Batch size: {agent.batch_size}")
    
    # Test action selection
    dummy_state = np.random.random(state_shape)
    action = agent.select_action(dummy_state)
    print(f"Selected action: {action}")
    
    # Test experience storage
    next_state = np.random.random(state_shape)
    agent.store_experience(dummy_state, action, 1.0, next_state, False)
    print(f"Memory size after storing: {len(agent.memory)}")
    
    # Test training (won't train until enough experiences)
    loss = agent.train_step()
    print(f"Training loss (should be None until enough data): {loss}")
    
    return agent

if __name__ == "__main__":
    test_agent()
