import numpy as np
from typing import Dict, List, Optional
import sys
import os

# Add src to path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from environment.network import PhysicalNetwork, VirtualNetwork

class FirstFit:
    """First Fit baseline algorithm - embeds in first available node"""
    def embed(self, physical_net: PhysicalNetwork, virtual_net: VirtualNetwork) -> Optional[Dict]:
        embedding = {}
        for vn_id, v_node in virtual_net.nodes.items():
            embedded = False
            for pn_id, p_node in physical_net.nodes.items():
                if p_node.available_cpu >= v_node.cpu_capacity:
                    embedding[vn_id] = pn_id
                    # Allocate resources
                    p_node.allocate_cpu(v_node.cpu_capacity)
                    embedded = True
                    break
            if not embedded:
                return None  # Failed to embed
        return embedding

class BestFit:
    """Best Fit baseline algorithm - embeds in node with least available capacity that fits"""
    def embed(self, physical_net: PhysicalNetwork, virtual_net: VirtualNetwork) -> Optional[Dict]:
        embedding = {}
        for vn_id, v_node in virtual_net.nodes.items():
            best_node = None
            best_remaining = float('inf')
            
            for pn_id, p_node in physical_net.nodes.items():
                remaining = p_node.available_cpu - v_node.cpu_capacity
                if remaining >= 0 and remaining < best_remaining:
                    best_node = pn_id
                    best_remaining = remaining
            
            if best_node is not None:
                embedding[vn_id] = best_node
                physical_net.nodes[best_node].allocate_cpu(v_node.cpu_capacity)
            else:
                return None  # Failed to embed
        return embedding

class RandomFit:
    """Random Fit baseline algorithm - embeds in random available node"""
    def embed(self, physical_net: PhysicalNetwork, virtual_net: VirtualNetwork) -> Optional[Dict]:
        embedding = {}
        available_nodes = list(physical_net.nodes.keys())
        
        for vn_id, v_node in virtual_net.nodes.items():
            embedded = False
            np.random.shuffle(available_nodes)  # Shuffle for random selection
            
            for pn_id in available_nodes:
                p_node = physical_net.nodes[pn_id]
                if p_node.available_cpu >= v_node.cpu_capacity:
                    embedding[vn_id] = pn_id
                    p_node.allocate_cpu(v_node.cpu_capacity)
                    embedded = True
                    break
            if not embedded:
                return None  # Failed to embed
        return embedding

def test_baselines():
    """Test all baseline algorithms"""
    from environment.vne_env import VNEEnvironment
    
    env = VNEEnvironment()
    vn = env.generate_virtual_network()
    
    algorithms = {
        'FirstFit': FirstFit(),
        'BestFit': BestFit(), 
        'RandomFit': RandomFit()
    }
    
    print("Baseline Algorithms Test:")
    print(f"Physical Network: {len(env.physical_network.nodes)} nodes")
    print(f"Virtual Network: {len(vn.nodes)} nodes, CPU demand: {vn.get_total_demand()['cpu']:.2f}")
    print("-" * 50)
    
    results = {}
    for name, algorithm in algorithms.items():
        # Create a fresh copy of physical network for each test
        test_env = VNEEnvironment()
        result = algorithm.embed(test_env.physical_network, vn)
        results[name] = result
        print(f"{name}: {'SUCCESS' if result else 'FAILED'}")
        
        if result:
            print(f"  Embedding: {result}")
            # Calculate resource utilization
            used_cpu = sum(vn.nodes[vid].cpu_capacity for vid in vn.nodes)
            total_cpu = sum(pn.cpu_capacity for pn in test_env.physical_network.nodes.values())
            util = (used_cpu / total_cpu) * 100
            print(f"  CPU Utilization: {util:.1f}%")
    
    return results

if __name__ == "__main__":
    test_baselines()
