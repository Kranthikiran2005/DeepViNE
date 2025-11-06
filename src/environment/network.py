import numpy as np
from typing import List, Dict, Tuple, Optional

class NetworkNode:
    """Base class for network nodes"""
    def __init__(self, node_id: int, cpu_capacity: float):
        self.id = node_id
        self.cpu_capacity = cpu_capacity
        self.available_cpu = cpu_capacity
        
    def allocate_cpu(self, demand: float) -> bool:
        """Allocate CPU if sufficient capacity available"""
        if demand <= self.available_cpu:
            self.available_cpu -= demand
            return True
        return False
    
    def release_cpu(self, demand: float):
        """Release allocated CPU"""
        self.available_cpu += demand

class NetworkLink:
    """Base class for network links"""
    def __init__(self, from_node: int, to_node: int, bandwidth_capacity: float):
        self.from_node = from_node
        self.to_node = to_node
        self.bandwidth_capacity = bandwidth_capacity
        self.available_bandwidth = bandwidth_capacity
        
    def allocate_bandwidth(self, demand: float) -> bool:
        """Allocate bandwidth if sufficient capacity available"""
        if demand <= self.available_bandwidth:
            self.available_bandwidth -= demand
            return True
        return False
    
    def release_bandwidth(self, demand: float):
        """Release allocated bandwidth"""
        self.available_bandwidth += demand

class PhysicalNetwork:
    """Physical network infrastructure"""
    def __init__(self):
        self.nodes: Dict[int, NetworkNode] = {}
        self.links: Dict[Tuple[int, int], NetworkLink] = {}
        self.grid_size = (0, 0)  # For grid topology
        
    def add_node(self, node_id: int, cpu_capacity: float):
        """Add a physical node"""
        self.nodes[node_id] = NetworkNode(node_id, cpu_capacity)
        
    def add_link(self, from_node: int, to_node: int, bandwidth_capacity: float):
        """Add a physical link"""
        link_id = (min(from_node, to_node), max(from_node, to_node))
        self.links[link_id] = NetworkLink(from_node, to_node, bandwidth_capacity)
        
    def get_available_resources(self) -> Dict:
        """Get current available resources"""
        return {
            'cpu': sum(node.available_cpu for node in self.nodes.values()),
            'bandwidth': sum(link.available_bandwidth for link in self.links.values())
        }

class VirtualNetwork:
    """Virtual network request"""
    def __init__(self, vn_id: int, arrival_time: float, lifetime: float):
        self.id = vn_id
        self.arrival_time = arrival_time
        self.lifetime = lifetime
        self.nodes: Dict[int, NetworkNode] = {}
        self.links: Dict[Tuple[int, int], NetworkLink] = {}
        self.embedding: Dict = {}  # Store embedding results
        
    def add_node(self, node_id: int, cpu_demand: float):
        """Add a virtual node"""
        self.nodes[node_id] = NetworkNode(node_id, cpu_demand)
        
    def add_link(self, from_node: int, to_node: int, bandwidth_demand: float):
        """Add a virtual link"""
        link_id = (min(from_node, to_node), max(from_node, to_node))
        self.links[link_id] = NetworkLink(from_node, to_node, bandwidth_demand)
        
    def get_total_demand(self) -> Dict:
        """Get total resource demands"""
        return {
            'cpu': sum(node.cpu_capacity for node in self.nodes.values()),
            'bandwidth': sum(link.bandwidth_capacity for link in self.links.values())
            #py
        }
