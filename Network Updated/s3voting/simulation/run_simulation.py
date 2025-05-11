import asyncio
import logging
import random
import time
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

from blockchain.node import BlockchainNode
from blockchain.shard import ShardManager
from blockchain.reputation import ReputationSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Simulation:
    def __init__(
        self,
        num_nodes: int = 10,
        num_shards: int = 5,
        malicious_ratio: float = 0.2,
        attack_pattern: str = "On-Off",
        enable_reshuffling: bool = True,
        num_voters: int = 100
    ):
        self.num_nodes = num_nodes
        self.num_malicious = int(num_nodes * malicious_ratio)
        self.shard_manager = ShardManager(num_shards)
        self.reputation_system = ReputationSystem()
        self.nodes: List[BlockchainNode] = []
        self.stop_event = asyncio.Event()
        self.reputation_history: List[Dict] = []
        self.malicious_distribution_history: List[Dict] = []
        self.performance_metrics: Dict[str, List[float]] = {
            "voters": [],
            "processing_time": [],
            "num_shards": [],
            "reputation_history": [],
            "malicious_distribution": []
        }
        self.start_time = time.time()
        self.malicious_nodes = []
        self.attack_pattern = attack_pattern
        self.enable_reshuffling = enable_reshuffling
        self.num_voters = num_voters
        self.voting_results = []  # Track voting results
        self.voters_vs_time = []  # Track (voters_processed, cumulative_time)

    def _get_attack_behavior(self, node_id: str) -> Dict:
        """Get attack behavior based on selected pattern"""
        if self.attack_pattern == "On-Off":
            return {"type": "on_off", "probability": 0.5}
        elif self.attack_pattern == "Persistent":
            return {"type": "persistent", "probability": 1.0}
        elif self.attack_pattern == "Sporadic":
            return {"type": "sporadic", "probability": 0.2}
        elif self.attack_pattern == "Targeted":
            return {"type": "targeted", "target_shard": random.randint(0, self.shard_manager.num_shards - 1)}
        else:
            return {"type": "on_off", "probability": 0.5}  # Default to on-off

    async def setup_nodes(self):
        """Create and start all nodes"""
        # Create malicious nodes
        self.malicious_nodes = [f"node_{i}" for i in range(self.num_malicious)]
        
        for i in range(self.num_nodes):
            node_id = f"node_{i}"
            port = 5000 + i
            shard_id = self.shard_manager.assign_node_to_shard(node_id)
            is_malicious = i < self.num_malicious
            
            node = BlockchainNode(
                node_id=node_id,
                shard_id=shard_id,
                port=port,
                is_malicious=is_malicious,
                reputation_system=self.reputation_system
            )
            self.nodes.append(node)
            await node.start()
            self.reputation_system.initialize_node(node_id)

    async def simulate_voting(self):
        """Simulate voting process"""
        logger.info(f"\n=== Starting Voting Simulation with {self.num_voters} voters ===")
        cumulative_time = 0.0
        for i in range(1, self.num_voters + 1):
            iteration_start = time.time()
            logger.info(f"\nVoting Iteration {i}/{self.num_voters}")
            voter_id = f"voter_{i}"
            candidate_id = f"candidate_{random.randint(1, 3)}"
            shard_id = random.randint(0, self.shard_manager.num_shards - 1)
            
            # Create vote transaction
            vote = {
                "type": "vote",
                "voter_id": voter_id,
                "candidate_id": candidate_id,
                "shard_id": shard_id
            }
            
            # Add vote to all nodes in the shard
            for node in self.nodes:
                if node.shard_id == shard_id:
                    await node.add_transaction(vote)
            
            # Track voting result
            self.voting_results.append({
                "voter_id": voter_id,
                "candidate_id": candidate_id,
                "shard_id": shard_id,
                "timestamp": time.time() - self.start_time
            })
            
            logger.info(f"Vote cast by {voter_id} for {candidate_id} (Shard {shard_id})")
            
            # Record malicious distribution
            current_time = time.time()
            if self.enable_reshuffling and self.shard_manager.should_reshuffle(current_time):
                self.shard_manager.reshuffle_shards(self.malicious_nodes)
                logger.info("Shards reshuffled")
            
            distribution = self.shard_manager.get_malicious_distribution()
            self.malicious_distribution_history.append({
                "time": current_time - self.start_time,
                "distribution": distribution
            })
            
            # Record performance metrics
            processing_time = time.time() - iteration_start
            self.performance_metrics["voters"].append(i)
            self.performance_metrics["processing_time"].append(processing_time)
            self.performance_metrics["num_shards"].append(self.shard_manager.num_shards)
            self.performance_metrics["reputation_history"].append(
                {node.node_id: self.reputation_system.get_reputation(node.node_id) for node in self.nodes}
            )
            self.performance_metrics["malicious_distribution"].append(distribution)
            
            # Track cumulative time for each vote
            cumulative_time += processing_time
            self.voters_vs_time.append((i, cumulative_time))
            
            # Update reputation every 10 votes
            if i % 10 == 0:
                await self.update_reputation()
                logger.info(f"\nProcessed {i} votes...")
                
            # Add small delay to allow UI updates
            await asyncio.sleep(0.1)

    async def run(self) -> Dict:
        """Run the simulation and return metrics"""
        logger.info("Starting blockchain simulation...")
        
        # Setup nodes
        await self.setup_nodes()
        logger.info(f"Created {self.num_nodes} nodes ({self.num_malicious} malicious)")
        
        # Connect peers
        await self.connect_peers()
        logger.info("Connected peers within shards")
        
        # Start mining process
        mining_tasks = [asyncio.create_task(node.start_mining()) for node in self.nodes]
        logger.info("Started mining process")
        
        # Start reputation updates
        reputation_task = asyncio.create_task(self.update_reputation())
        logger.info("Started reputation updates")
        
        # Simulate voting
        await self.simulate_voting()
        
        # Set stop event after 60 seconds
        await asyncio.sleep(60)
        self.stop_event.set()
        
        # Stop all nodes
        for node in self.nodes:
            await node.stop()
            
        # Cancel reputation updates and mining tasks
        reputation_task.cancel()
        for task in mining_tasks:
            task.cancel()
            
        # Wait for all tasks to complete
        await asyncio.gather(*mining_tasks, return_exceptions=True)
        await asyncio.gather(reputation_task, return_exceptions=True)
        
        # Calculate final metrics
        final_metrics = self.performance_metrics.copy()
        final_metrics.update({
            "total_nodes": self.num_nodes,
            "malicious_nodes": self.num_malicious,
            "attack_pattern": self.attack_pattern,
            "enable_reshuffling": self.enable_reshuffling,
            "final_reputation": {node.node_id: self.reputation_system.get_reputation(node.node_id) 
                               for node in self.nodes},
            "final_malicious_distribution": self.shard_manager.get_malicious_distribution(),
            "voting_results": self.voting_results,  # Add voting results to metrics
            "voters_vs_time": self.voters_vs_time  # Add voters vs time to metrics
        })
        
        return final_metrics

    async def connect_peers(self):
        """Connect nodes within the same shard"""
        for node in self.nodes:
            shard_nodes = self.shard_manager.get_nodes_in_shard(node.shard_id)
            for peer_id in shard_nodes:
                if peer_id != node.node_id:
                    peer = next(n for n in self.nodes if n.node_id == peer_id)
                    await node.add_peer(peer_id, peer.host, peer.port)

    async def update_reputation(self):
        """Periodically update reputation scores"""
        iteration = 0
        start_time = time.time()
        while True:
            try:
                if time.time() - start_time >= 60:  # Stop after 60 seconds
                    break
                iteration += 1
                logger.info(f"\nReputation Update Iteration {iteration}")
                
                # Collect behavior data from all nodes
                behavior_data = {}
                for node in self.nodes:
                    behavior = node.get_current_behavior()
                    if behavior:
                        behavior_data[node.node_id] = behavior
                
                # Update reputations
                self.reputation_system.update_reputations(behavior_data)
                
                # Log trusted nodes
                trusted_nodes = self.reputation_system.get_trusted_nodes()
                logger.info(f"Trusted nodes: {trusted_nodes}")
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error updating reputation: {str(e)}")
                await asyncio.sleep(5)

    def print_final_state(self):
        """Print the final state of the simulation"""
        logger.info("\n=== Final Simulation State ===")
        
        # Print shard statistics
        logger.info("\nShard Statistics:")
        shard_stats = self.shard_manager.get_shard_stats()
        for shard_id, stats in shard_stats.items():
            logger.info(f"Shard {shard_id}:")
            logger.info(f"  Number of nodes: {stats['num_nodes']}")
            logger.info(f"  Nodes: {', '.join(stats['nodes'])}")
        
        # Print blockchain state for each node
        logger.info("\nBlockchain State:")
        for node in self.nodes:
            state = node.get_chain_state()
            rep = self.reputation_system.get_reputation(node.node_id)
            logger.info(f"\nNode {state['node_id']} (Shard {state['shard_id']}):")
            logger.info(f"  Chain length: {state['chain_length']}")
            logger.info(f"  Pending transactions: {state['pending_transactions']}")
            logger.info(f"  Number of peers: {state['peers']}")
            logger.info(f"  Is malicious: {state['is_malicious']}")
            logger.info(f"  Reputation: {rep:.3f}")
            logger.info(f"  Is trusted: {self.reputation_system.is_trusted(node.node_id)}")

    def plot_results(self):
        """Plot simulation results"""
        # Create directory for results if it doesn't exist
        import os
        os.makedirs("simulation_results", exist_ok=True)
        
        # Plot malicious node distribution
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Malicious Node Distribution Over Time
        plt.subplot(2, 2, 1)
        if self.malicious_distribution_history:
            df_dist = pd.DataFrame(self.malicious_distribution_history)
            for shard_id in range(self.shard_manager.num_shards):
                shard_data = [d["distribution"].get(shard_id, 0) for d in self.malicious_distribution_history]
                plt.plot(df_dist["time"], shard_data, label=f"Shard {shard_id}")
            plt.title("Malicious Node Distribution Over Time")
            plt.xlabel("Time (seconds)")
            plt.ylabel("Number of Malicious Nodes")
            plt.legend()
            plt.grid(True)
        
        # Plot 2: Processing Time vs Number of Voters
        plt.subplot(2, 2, 2)
        df_perf = pd.DataFrame(self.performance_metrics)
        plt.plot(df_perf["voters"], df_perf["processing_time"], 'b-', label='Processing Time')
        plt.title("Processing Time vs Number of Voters")
        plt.xlabel("Number of Voters")
        plt.ylabel("Processing Time (seconds)")
        plt.grid(True)
        
        # Plot 3: Reputation Evolution
        plt.subplot(2, 2, 3)
        if self.reputation_history:
            df_rep = pd.DataFrame(self.reputation_history)
            plt.plot(df_rep["time"], df_rep["mean_reputation"], 'g-', label='Mean Reputation')
            plt.title("Reputation Evolution")
            plt.xlabel("Time")
            plt.ylabel("Mean Reputation")
            plt.grid(True)
        
        # Plot 4: Trusted Nodes Over Time
        plt.subplot(2, 2, 4)
        if self.reputation_history:
            plt.plot(df_rep["time"], df_rep["trusted_nodes"], 'r-', label='Trusted Nodes')
            plt.title("Trusted Nodes Over Time")
            plt.xlabel("Time")
            plt.ylabel("Number of Trusted Nodes")
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('simulation_results/simulation_results.png')
        plt.close()
        
        # Save detailed metrics
        self.save_metrics()

    def save_metrics(self):
        """Save detailed simulation metrics"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save malicious distribution data
        if self.malicious_distribution_history:
            df_dist = pd.DataFrame(self.malicious_distribution_history)
            df_dist.to_csv(f'simulation_results/malicious_distribution_{timestamp}.csv', index=False)
        
        # Save performance metrics
        df_perf = pd.DataFrame(self.performance_metrics)
        df_perf.to_csv(f'simulation_results/performance_metrics_{timestamp}.csv', index=False)
        
        # Save reputation history
        if self.reputation_history:
            df_rep = pd.DataFrame(self.reputation_history)
            df_rep.to_csv(f'simulation_results/reputation_history_{timestamp}.csv', index=False)

    def print_blockchain_state(self):
        """Print the final state of the blockchain"""
        logger.info("\n=== Final Blockchain State ===")
        
        # Print blockchain state for each node
        logger.info("\nBlockchain State:")
        for node in self.nodes:
            state = node.get_chain_state()
            logger.info(f"\nNode {state['node_id']} (Shard {state['shard_id']}):")
            logger.info(f"  Chain length: {state['chain_length']}")
            logger.info(f"  Pending transactions: {state['pending_transactions']}")
            logger.info(f"  Number of peers: {state['peers']}")
            logger.info(f"  Is malicious: {state['is_malicious']}")

async def main():
    """Main entry point for running the simulation directly"""
    sim = Simulation()
    metrics = await sim.run()
    print("Simulation completed!")
    print("Metrics:", metrics)

if __name__ == "__main__":
    asyncio.run(main()) 