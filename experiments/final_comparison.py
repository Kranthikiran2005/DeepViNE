# Create experiments/final_comparison.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.agents.deepvine_agent import DeepViNEAgent
from src.algorithms.baselines import FirstFit, BestFit, RandomFit
from src.environment.vne_env import VNEEnvironment
import numpy as np
from tqdm import tqdm

def final_comparison():
    """Final comprehensive comparison"""
    print("FINAL COMPREHENSIVE COMPARISON")
    print("="*50)
    
    # Load your best trained model
    state_shape = (8, 8, 3)
    num_actions = 9
    deepvine_agent = DeepViNEAgent(state_shape, num_actions)
    deepvine_agent.load_model('./results/models/deepvine_best.pth')
    
    algorithms = {
        'DeepViNE (Trained)': deepvine_agent,
        'FirstFit': FirstFit(),
        'BestFit': BestFit(),
        'RandomFit': RandomFit()
    }
    
    results = {}
    num_tests = 100
    
    for algo_name, algorithm in algorithms.items():
        print(f"\nTesting {algo_name}...")
        acceptance_rates = []
        revenues = []
        steps_taken = []
        
        for test in tqdm(range(num_tests), desc=f"Testing {algo_name}"):
            env = VNEEnvironment(difficulty="mixed")
            
            if algo_name == 'DeepViNE (Trained)':
                state = env._get_state()
                total_reward = 0
                done = False
                steps = 0
                
                while not done and steps < 50:
                    action = algorithm.select_action(state)
                    next_state, reward, done, info = env.step(action)
                    state = next_state
                    total_reward += reward
                    steps += 1
                
                acceptance_rates.append(info['acceptance_rate'])
                revenues.append(info['total_revenue'])
                steps_taken.append(steps)
            else:
                result = algorithm.embed(env.physical_network, env.current_vn)
                acceptance_rate = 1.0 if result else 0.0
                revenue = sum(env.current_vn.nodes[vid].cpu_capacity for vid in env.current_vn.nodes) if result else 0
                
                acceptance_rates.append(acceptance_rate)
                revenues.append(revenue)
                steps_taken.append(1)  # Baselines are instant
        
        results[algo_name] = {
            'avg_acceptance_rate': np.mean(acceptance_rates),
            'avg_revenue': np.mean(revenues),
            'avg_steps': np.mean(steps_taken),
            'std_acceptance': np.std(acceptance_rates)
        }
        
        print(f"  Acceptance Rate: {results[algo_name]['avg_acceptance_rate']:.3f}")
        print(f"  Average Revenue: {results[algo_name]['avg_revenue']:.3f}")
        print(f"  Average Steps: {results[algo_name]['avg_steps']:.1f}")
    
    # Print final comparison
    print("\n" + "="*60)
    print("FINAL COMPARISON RESULTS FOR PROJECT REPORT")
    print("="*60)
    for algo_name, metrics in results.items():
        print(f"\n{algo_name}:")
        print(f"  • Acceptance Rate: {metrics['avg_acceptance_rate']:.3f} ± {metrics['std_acceptance']:.3f}")
        print(f"  • Average Revenue: {metrics['avg_revenue']:.3f}")
        print(f"  • Steps Required: {metrics['avg_steps']:.1f}")
    
    return results

if __name__ == "__main__":
    final_comparison()