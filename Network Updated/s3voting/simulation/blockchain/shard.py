import hashlib
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class ShardManager:
    def __init__(self, num_shards: int = 5):
        self.num_shards = num_shards
        self.node_shards: Dict[str, int] = {}  # node_id -> shard_id
        self.shard_nodes: Dict[int, List[str]] = {i: [] for i in range(num_shards)}  # shard_id -> [node_ids]

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