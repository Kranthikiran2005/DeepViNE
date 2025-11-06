import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.agents.deepvine_agent import DeepViNEAgent
from src.environment.vne_env import VNEEnvironment

def train_improved(episodes=1000):
    """Improved training with better hyperparameters"""
    print("Starting Improved DeepViNE Training...")
    
    # Initialize environment and agent with better settings
    # Use mixed difficulty for more challenging problems
    env = VNEEnvironment(grid_size=(5, 5), difficulty="mixed")
    state_shape = (8, 8, 3)
    num_actions = 9
    
    # Use standard agent but we'll modify hyperparameters after creation
    agent = DeepViNEAgent(state_shape, num_actions)
    
    # Modify hyperparameters for better training
    agent.learning_rate = 0.0005  # Lower learning rate for stability
    agent.gamma = 0.95           # Slightly lower discount factor
    agent.epsilon_decay = 0.998  # Slower epsilon decay
    agent.epsilon_min = 0.01     # Minimum exploration
    
    # Recreate optimizer with new learning rate
    agent.optimizer = torch.optim.Adam(agent.policy_net.parameters(), lr=agent.learning_rate)
    
    # Training metrics
    episode_rewards = []
    episode_acceptance_rates = []
    losses = []
    best_reward = -float('inf')
    
    for episode in tqdm(range(episodes), desc="Training Episodes"):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 100:  # Limit steps per episode
            # Select action
            action = agent.select_action(state)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            agent.store_experience(state, action, reward, next_state, done)
            
            # Train agent more frequently
            if len(agent.memory) > agent.batch_size:
                loss = agent.train_step()
                if loss is not None:
                    losses.append(loss)
            
            state = next_state
            total_reward += reward
            steps += 1
        
        # Record metrics
        episode_rewards.append(total_reward)
        episode_acceptance_rates.append(info['acceptance_rate'])
        
        # Save best model
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save_model('./results/models/deepvine_best.pth')
        
        # Print progress every 50 episodes
        if episode % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.mean(episode_rewards)
            avg_acceptance = np.mean(episode_acceptance_rates[-50:]) if len(episode_acceptance_rates) >= 50 else np.mean(episode_acceptance_rates)
            print(f"\nEpisode {episode}: Avg Reward={avg_reward:.3f}, Avg Acceptance={avg_acceptance:.3f}, Epsilon={agent.epsilon:.3f}")
    os.makedirs('./results/models',exist_ok=True)
    agent.save_model('./results/models/deepvine_best.pth')
    # Create plots directory
    os.makedirs('./results/plots', exist_ok=True)
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    plt.subplot(1, 3, 2)
    plt.plot(episode_acceptance_rates)
    plt.title('Acceptance Rate')
    plt.xlabel('Episode')
    plt.ylabel('Acceptance Rate')
    
    plt.subplot(1, 3, 3)
    if losses:
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig('./results/plots/improved_training_results.png')
    plt.show()
    
    return agent, episode_rewards, episode_acceptance_rates

if __name__ == "__main__":
    trained_agent, rewards, acceptance_rates = train_improved(episodes=1000)
    print("Improved training completed!")