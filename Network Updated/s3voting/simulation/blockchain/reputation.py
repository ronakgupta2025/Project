import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from numpy import tanh
import logging
from typing import Dict, List, Tuple, Set
import time
import matplotlib.pyplot as plt
import seaborn as sns

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
        self.scaler = StandardScaler()
        self.attack_patterns = {
            "on_off": [],
            "gradual": [],
            "burst": [],
            "adaptive": []
        }
        
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
        
    def update_reputations(self, behavior_data: Dict[str, Dict]):
        """Update reputations based on behavior data"""
        for node_id, behavior in behavior_data.items():
            if node_id not in self.behavior_history:
                self.initialize_node(node_id)
                
            # Record behavior
            self.behavior_history[node_id].append(behavior)
            
            # Keep only last 100 behaviors
            if len(self.behavior_history[node_id]) > 100:
                self.behavior_history[node_id] = self.behavior_history[node_id][-100:]
                
            # Detect anomalies using Isolation Forest
            if len(self.behavior_history[node_id]) >= 10:
                features = self._extract_features(node_id)
                if features is not None:
                    anomaly_score = self._detect_anomalies(features)
                    self._update_reputation(node_id, behavior, anomaly_score)
                    
                    # Record attack patterns
                    if behavior.get("attack_type") != "honest":
                        self.attack_patterns[behavior["attack_type"]].append({
                            "node_id": node_id,
                            "time": len(self.behavior_history[node_id]),
                            "anomaly_score": anomaly_score,
                            "reputation": self.reputations[node_id]
                        })
    
    def _extract_features(self, node_id: str) -> np.ndarray:
        """Extract features from behavior history for anomaly detection"""
        history = self.behavior_history[node_id]
        if len(history) < 10:
            return None
            
        features = []
        for behavior in history[-10:]:
            features.append([
                behavior["block_delay"],
                behavior["voting_mismatch"],
                behavior["penalties"]
            ])
        return np.array(features)
    
    def _detect_anomalies(self, features: np.ndarray) -> float:
        """Detect anomalies using Isolation Forest"""
        features_scaled = self.scaler.fit_transform(features)
        return -self.isolation_forest.fit_predict(features_scaled).mean()
    
    def _update_reputation(self, node_id: str, behavior: Dict, anomaly_score: float):
        """Update node reputation based on behavior and anomaly score"""
        current_rep = self.reputations[node_id]
        
        # Calculate reputation change
        delay_penalty = max(0, (behavior["block_delay"] - 2.0) / 10.0)
        mismatch_penalty = behavior["voting_mismatch"]
        penalty_factor = behavior["penalties"] * 0.1
        anomaly_penalty = max(0, anomaly_score)
        
        total_penalty = (delay_penalty + mismatch_penalty + penalty_factor + anomaly_penalty) / 4
        new_rep = max(0.1, current_rep - total_penalty)
        
        # Gradual recovery for honest behavior
        if not behavior["is_attack_phase"]:
            new_rep = min(1.0, new_rep + 0.05)
            
        self.reputations[node_id] = new_rep
        logger.info(f"Node {node_id} reputation updated: {current_rep:.3f} -> {new_rep:.3f}")
    
    def get_reputation(self, node_id: str) -> float:
        """Get current reputation of a node"""
        return self.reputations.get(node_id, 1.0)
        
    def get_mean_reputation(self) -> float:
        """Get the mean reputation score across all nodes"""
        if not self.reputations:
            return 1.0
        return sum(self.reputations.values()) / len(self.reputations)
        
    def get_trusted_nodes(self) -> Set[str]:
        """Get set of trusted nodes"""
        return {node_id for node_id, rep in self.reputations.items() 
                if rep >= self.trust_threshold}
        
    def is_trusted(self, node_id: str) -> bool:
        """Check if a node is trusted based on reputation"""
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
        
    def plot_attack_patterns(self, output_file: str = "attack_patterns.png"):
        """Plot attack patterns and their detection"""
        plt.figure(figsize=(15, 10))
        
        # Plot reputation evolution for each attack type
        for attack_type, patterns in self.attack_patterns.items():
            if patterns:
                df = pd.DataFrame(patterns)
                plt.subplot(2, 2, list(self.attack_patterns.keys()).index(attack_type) + 1)
                plt.plot(df["time"], df["reputation"], label="Reputation")
                plt.plot(df["time"], df["anomaly_score"], label="Anomaly Score")
                plt.title(f"{attack_type.capitalize()} Attack Pattern")
                plt.xlabel("Time")
                plt.ylabel("Score")
                plt.legend()
                plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        
    def save_behavior_metrics(self, output_file: str = "behavior_metrics.txt"):
        """Save detailed behavior metrics to file"""
        with open(output_file, "w") as f:
            f.write("Node Behavior Metrics\n")
            f.write("===================\n\n")
            
            for node_id, history in self.behavior_history.items():
                f.write(f"\nNode {node_id}:\n")
                f.write("-" * 50 + "\n")
                
                # Calculate statistics
                delays = [b["block_delay"] for b in history]
                mismatches = [b["voting_mismatch"] for b in history]
                penalties = [b["penalties"] for b in history]
                attack_types = [b.get("attack_type", "honest") for b in history]
                
                f.write(f"Current Reputation: {self.reputations[node_id]:.3f}\n")
                f.write(f"Trusted: {self.is_trusted(node_id)}\n")
                f.write(f"Attack Types: {', '.join(set(attack_types))}\n")
                f.write(f"Average Block Delay: {np.mean(delays):.2f}\n")
                f.write(f"Average Voting Mismatch: {np.mean(mismatches):.2f}\n")
                f.write(f"Total Penalties: {sum(penalties)}\n")
                f.write(f"Number of Behaviors: {len(history)}\n")
                f.write("\n") 