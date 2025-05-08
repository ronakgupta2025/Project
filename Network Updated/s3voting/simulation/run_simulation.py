import asyncio
import logging
import random
import time
from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np

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
    def __init__(self, num_nodes: int = 10, num_shards: int = 5, malicious_ratio: float = 0.2):
        self.num_nodes = num_nodes
        self.num_malicious = int(num_nodes * malicious_ratio)
        self.shard_manager = ShardManager(num_shards)
        self.reputation_system = ReputationSystem()
        self.nodes: List[BlockchainNode] = []
        self.stop_event = asyncio.Event()
        self.reputation_history: List[Dict] = []

    async def setup_nodes(self):
        """Create and start all nodes"""
        # Create malicious nodes
        malicious_nodes = random.sample(range(self.num_nodes), self.num_malicious)
        
        for i in range(self.num_nodes):
            node_id = f"node_{i}"
            port = 5000 + i
            shard_id = self.shard_manager.assign_node_to_shard(node_id)
            is_malicious = i in malicious_nodes
            node = BlockchainNode(node_id, shard_id, port=port, is_malicious=is_malicious)
            self.nodes.append(node)
            await node.start()
            self.reputation_system.initialize_node(node_id)

    async def connect_peers(self):
        """Connect nodes within the same shard"""
        for node in self.nodes:
            shard_nodes = self.shard_manager.get_nodes_in_shard(node.shard_id)
            for peer_id in shard_nodes:
                if peer_id != node.node_id:
                    peer = next(n for n in self.nodes if n.node_id == peer_id)
                    await node.add_peer(peer_id, peer.host, peer.port)

    async def simulate_voting(self):
        """Simulate voting process"""
        logger.info("\n=== Starting Voting Simulation ===")
        
        # Generate more transactions
        num_voters = 100  # Increased from 30
        for i in range(1, num_voters + 1):
            voter_id = f"voter_{i}"
            candidate_id = f"candidate_{random.randint(1, 3)}"
            shard_id = random.randint(0, 4)
            
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
                    
            logger.info(f"Vote cast by {voter_id} for {candidate_id} (Shard {shard_id})")
            
            # Add small delay between votes
            await asyncio.sleep(0.1)
            
            # Update reputation every 10 votes
            if i % 10 == 0:
                await self.update_reputation()
                logger.info(f"\nProcessed {i} votes...")

    async def mine_blocks(self):
        """Mine blocks for all nodes"""
        while not self.stop_event.is_set():
            for node in self.nodes:
                if node.pending_transactions:
                    await node.mine_block()
                    # Record behavior for reputation system
                    behavior = node._simulate_malicious_behavior()
                    self.reputation_system.record_behavior(node.node_id, behavior)
            await asyncio.sleep(1)

    async def update_reputation(self):
        """Update reputation scores and log statistics"""
        # Collect behavior data from all nodes
        behavior_data = []
        for node in self.nodes:
            behavior = node._simulate_malicious_behavior()
            behavior_data.append({
                "node_id": node.node_id,
                "block_delay": behavior["block_delay"],
                "voting_mismatch": behavior["voting_mismatch"],
                "penalties": behavior["penalties"],
                "is_attack_phase": behavior.get("is_attack_phase", False)
            })
            
        # Update reputation scores
        self.reputation_system.update_reputations(behavior_data)
        
        # Log reputation statistics
        logger.info("\nReputation Statistics:")
        mean_rep = self.reputation_system.get_mean_reputation()
        trusted_nodes = self.reputation_system.get_trusted_nodes()
        logger.info(f"Mean Reputation: {mean_rep:.3f}")
        logger.info(f"Trusted Nodes: {len(trusted_nodes)}/{len(self.nodes)}")
        
        # Log detailed node states
        logger.info("\nDetailed Node States:")
        for node in self.nodes:
            rep = self.reputation_system.get_reputation(node.node_id)
            is_trusted = node.node_id in trusted_nodes
            behavior = node._simulate_malicious_behavior()
            logger.info(f"Node {node.node_id}:")
            logger.info(f"  Reputation: {rep:.3f}")
            logger.info(f"  Is Trusted: {is_trusted}")
            logger.info(f"  Attack Phase: {behavior.get('is_attack_phase', False)}")
            logger.info(f"  Block Delay: {behavior['block_delay']:.2f}")
            logger.info(f"  Voting Mismatch: {behavior['voting_mismatch']:.2f}")
            logger.info(f"  Penalties: {behavior['penalties']}")

    async def run(self):
        """Run the complete simulation"""
        try:
            # Setup and start nodes
            logger.info("Setting up nodes...")
            await self.setup_nodes()
            
            # Connect peers within shards
            logger.info("Connecting peers...")
            await self.connect_peers()
            
            # Start mining process
            mining_task = asyncio.create_task(self.mine_blocks())
            
            # Start reputation updates
            reputation_task = asyncio.create_task(self.update_reputation())
            
            # Simulate voting
            await self.simulate_voting()
            
            # Wait for some time to process votes
            await asyncio.sleep(10)
            
            # Stop mining and reputation updates
            self.stop_event.set()
            await mining_task
            await reputation_task
            
            # Print final state
            self.print_final_state()
            
            # Plot results
            self.plot_results()
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            raise
        finally:
            # Stop all nodes
            for node in self.nodes:
                await node.stop()

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
        # Plot reputation history
        plt.figure(figsize=(12, 6))
        epochs = range(len(self.reputation_history))
        
        mean_reps = [stats['mean_reputation'] for stats in self.reputation_history]
        trusted_nodes = [stats['trusted_nodes'] for stats in self.reputation_history]
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs, mean_reps, label='Mean Reputation')
        plt.xlabel('Time')
        plt.ylabel('Mean Reputation')
        plt.title('Reputation Evolution')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs, trusted_nodes, label='Trusted Nodes')
        plt.xlabel('Time')
        plt.ylabel('Number of Trusted Nodes')
        plt.title('Trusted Nodes Over Time')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('simulation_results.png')
        plt.close()

async def main():
    """Main entry point"""
    simulation = Simulation(num_nodes=10, num_shards=5, malicious_ratio=0.2)
    await simulation.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nSimulation interrupted by user")
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise 