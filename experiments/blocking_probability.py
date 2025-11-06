import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
import numpy as np
from src.agents.deepvine_agent import DeepViNEAgent
from src.algorithms.baselines import FirstFit, BestFit, RandomFit
from src.environment.vne_env import VNEEnvironment

def plot_blocking_probability():
    """Plot blocking probability comparison"""
    print("Generating Blocking Probability Comparison...")
    
    # Load your trained model
    state_shape = (8, 8, 3)
    num_actions = 9
    deepvine_agent = DeepViNEAgent(state_shape, num_actions)
    deepvine_agent.load_model('../results/models/deepvine_best.pth')
    
    algorithms = {
        'DeepViNE': deepvine_agent,
        'FirstFit': FirstFit(),
        'BestFit': BestFit(),
        'RandomFit': RandomFit()
    }
    
    blocking_rates = {}
    num_tests = 100
    
    for algo_name, algorithm in algorithms.items():
        print(f"Testing {algo_name}...")
        blocked_count = 0
        
        for test in range(num_tests):
            env = VNEEnvironment(difficulty="mixed")
            
            if algo_name == 'DeepViNE':
                state = env._get_state()
                done = False
                while not done:
                    action = algorithm.select_action(state)
                    next_state, reward, done, info = env.step(action)
                    state = next_state
                if info['blocked_vns'] > 0:
                    blocked_count += 1
            else:
                result = algorithm.embed(env.physical_network, env.current_vn)
                if not result:
                    blocked_count += 1
        
        blocking_rate = blocked_count / num_tests
        blocking_rates[algo_name] = blocking_rate
        print(f"  {algo_name}: {blocking_rate:.3f} blocking probability")
    
    # Create blocking probability plot
    plt.figure(figsize=(10, 6))
    algorithms_names = list(blocking_rates.keys())
    blocking_values = list(blocking_rates.values())
    
    bars = plt.bar(algorithms_names, blocking_values, color=['blue', 'orange', 'green', 'red'])
    plt.title('Virtual Network Blocking Probability Comparison')
    plt.ylabel('Blocking Probability')
    plt.ylim(0, 1.0)
    
    # Add value labels on bars
    for bar, value in zip(bars, blocking_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('../results/plots/blocking_probability.png')
    plt.show()
    
    return blocking_rates

if __name__ == "__main__":
    plot_blocking_probability()
