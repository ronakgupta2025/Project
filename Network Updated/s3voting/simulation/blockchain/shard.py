import hashlib
import logging
import random
import time
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class ShardManager:
    def __init__(self, num_shards: int = 5):
        self.num_shards = num_shards
        self.node_shards: Dict[str, int] = {}  # node_id -> shard_id
        self.shard_nodes: Dict[int, List[str]] = {i: [] for i in range(num_shards)}  # shard_id -> [node_ids]
        self.last_reshuffle_time = 0
        self.reshuffle_interval = 300  # 5 minutes in seconds

    def assign_node_to_shard(self, node_id: str) -> int:
        """Assign a node to a shard using consistent hashing"""
        if node_id in self.node_shards:
            return self.node_shards[node_id]

        # Use consistent hashing to assign shard
        hash_value = int(hashlib.sha256(node_id.encode()).hexdigest(), 16)
        shard_id = hash_value % self.num_shards

        self.node_shards[node_id] = shard_id
        self.shard_nodes[shard_id].append(node_id)
        logger.info(f"Assigned node {node_id} to shard {shard_id}")
        return shard_id

    def get_shard_for_vote(self, voter_id: str) -> int:
        """Get the shard ID for a voter using consistent hashing"""
        hash_value = int(hashlib.sha256(voter_id.encode()).hexdigest(), 16)
        return hash_value % self.num_shards

    def get_nodes_in_shard(self, shard_id: int) -> List[str]:
        """Get all nodes in a specific shard"""
        return self.shard_nodes.get(shard_id, [])

    def get_node_shard(self, node_id: str) -> Optional[int]:
        """Get the shard ID for a specific node"""
        return self.node_shards.get(node_id)

    def get_shard_stats(self) -> Dict[int, Dict]:
        """Get statistics for each shard"""
        stats = {}
        for shard_id in range(self.num_shards):
            stats[shard_id] = {
                "num_nodes": len(self.shard_nodes[shard_id]),
                "nodes": self.shard_nodes[shard_id]
            }
        return stats

    def should_reshuffle(self, current_time: float) -> bool:
        """Check if it's time to reshuffle shards"""
        return current_time - self.last_reshuffle_time >= self.reshuffle_interval

    def reshuffle_shards(self, malicious_nodes: List[str]) -> None:
        """Reshuffle nodes across shards to balance malicious and honest nodes"""
        logger.info("Starting shard reshuffling...")
        
        # Clear current assignments
        self.node_shards.clear()
        for shard_id in range(self.num_shards):
            self.shard_nodes[shard_id] = []

        # Get all nodes
        all_nodes = list(set().union(*[set(nodes) for nodes in self.shard_nodes.values()]))
        
        # Separate malicious and honest nodes
        honest_nodes = [node for node in all_nodes if node not in malicious_nodes]
        
        # Calculate target distribution
        nodes_per_shard = len(all_nodes) // self.num_shards
        malicious_per_shard = len(malicious_nodes) // self.num_shards
        
        # Distribute malicious nodes first
        for i, malicious_node in enumerate(malicious_nodes):
            target_shard = i % self.num_shards
            self.node_shards[malicious_node] = target_shard
            self.shard_nodes[target_shard].append(malicious_node)
        
        # Distribute honest nodes
        for i, honest_node in enumerate(honest_nodes):
            # Find shard with least nodes
            target_shard = min(range(self.num_shards), 
                             key=lambda x: len(self.shard_nodes[x]))
            self.node_shards[honest_node] = target_shard
            self.shard_nodes[target_shard].append(honest_node)
        
        self.last_reshuffle_time = time.time()
        logger.info("Shard reshuffling completed")
        
    def get_malicious_distribution(self) -> Dict[int, int]:
        """Get the distribution of malicious nodes across shards"""
        distribution = {}
        for shard_id in range(self.num_shards):
            malicious_count = sum(1 for node in self.shard_nodes[shard_id] 
                                if node.startswith("node_") and int(node.split("_")[1]) < 4)
            distribution[shard_id] = malicious_count 