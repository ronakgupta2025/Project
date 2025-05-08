import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from numpy import tanh
import logging
from typing import Dict, List, Tuple, Set
import time

logger = logging.getLogger(__name__)

class ReputationSystem:
    def __init__(self, contamination: float = 0.1):
        self.reputations: Dict[str, float] = {}
        self.behavior_history: Dict[str, List[Dict]] = {}
        self.contamination = contamination
        self.trust_threshold = 0.7
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42
        )
        
    def initialize_node(self, node_id: str):
        """Initialize a new node with default reputation"""
        self.reputations[node_id] = 1.0
        self.behavior_history[node_id] = []
        
    def record_behavior(self, node_id: str, behavior: Dict):
        """Record node behavior metrics"""
        if node_id not in self.behavior_history:
            self.initialize_node(node_id)
            
        behavior['timestamp'] = time.time()
        self.behavior_history[node_id].append(behavior)
        
    def _calculate_metrics(self, node_id: str) -> Dict:
        """Calculate behavior metrics for a node"""
        if not self.behavior_history[node_id]:
            return {
                "block_delay": 0.0,
                "voting_mismatch": 0.0,
                "penalties": 0
            }
            
        recent_behaviors = self.behavior_history[node_id][-10:]  # Look at last 10 behaviors
        
        # Calculate metrics
        block_delays = [b.get('block_delay', 0) for b in recent_behaviors]
        voting_mismatches = [b.get('voting_mismatch', 0) for b in recent_behaviors]
        penalties = [b.get('penalties', 0) for b in recent_behaviors]
        
        return {
            "block_delay": np.mean(block_delays) if block_delays else 0.0,
            "voting_mismatch": np.mean(voting_mismatches) if voting_mismatches else 0.0,
            "penalties": sum(penalties)
        }
        
    def update_reputations(self, behavior_data: List[Dict]):
        """Update reputation scores based on behavior data"""
        if not behavior_data:
            return
            
        # Extract features for anomaly detection
        features = []
        node_ids = []
        for data in behavior_data:
            features.append([
                data["block_delay"],
                data["voting_mismatch"],
                data["penalties"]
            ])
            node_ids.append(data["node_id"])
            
        # Convert to numpy array
        X = np.array(features)
        
        # Fit and predict anomalies
        try:
            self.isolation_forest.fit(X)
            anomaly_scores = self.isolation_forest.score_samples(X)
            
            # Update reputations based on anomaly scores
            for i, node_id in enumerate(node_ids):
                # Normalize anomaly score to [0, 1] range
                normalized_score = (anomaly_scores[i] - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())
                
                # Update reputation with exponential moving average
                current_rep = self.reputations.get(node_id, 1.0)
                new_rep = 0.7 * current_rep + 0.3 * normalized_score
                self.reputations[node_id] = max(0.0, min(1.0, new_rep))
                
                # Log reputation changes
                logger.info(f"Node {node_id} reputation updated: {current_rep:.3f} -> {new_rep:.3f}")
                
        except Exception as e:
            logger.error(f"Error updating reputations: {str(e)}")
            
    def get_reputation(self, node_id: str) -> float:
        """Get the current reputation score for a node"""
        return self.reputations.get(node_id, 1.0)
        
    def get_mean_reputation(self) -> float:
        """Get the mean reputation score across all nodes"""
        if not self.reputations:
            return 1.0
        return sum(self.reputations.values()) / len(self.reputations)
        
    def get_trusted_nodes(self) -> Set[str]:
        """Get the set of trusted nodes based on reputation threshold"""
        return {node_id for node_id, rep in self.reputations.items() if rep >= self.trust_threshold}
        
    def is_trusted(self, node_id: str) -> bool:
        """Check if a node is trusted based on its reputation"""
        return self.get_reputation(node_id) >= self.trust_threshold
        
    def get_reputation_stats(self) -> Dict:
        """Get statistics about the reputation system"""
        if not self.reputations:
            return {}
            
        reputations = list(self.reputations.values())
        return {
            "mean_reputation": np.mean(reputations),
            "std_reputation": np.std(reputations),
            "min_reputation": np.min(reputations),
            "max_reputation": np.max(reputations),
            "trusted_nodes": len(self.get_trusted_nodes()),
            "total_nodes": len(self.reputations)
        } 