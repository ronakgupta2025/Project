import asyncio
import hashlib
import json
import logging
import time
import random
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Block:
    index: int
    timestamp: float
    transactions: List[Dict]
    previous_hash: str
    hash: str = ""
    shard_id: int = 0

    def __post_init__(self):
        if not self.hash:
            self.hash = self.calculate_hash()

    def calculate_hash(self) -> str:
        block_string = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "transactions": self.transactions,
            "previous_hash": self.previous_hash,
            "shard_id": self.shard_id
        }, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()

class BlockchainNode:
    def __init__(self, node_id: str, shard_id: int, host: str = "127.0.0.1", port: int = 5000, is_malicious: bool = False, reputation_system = None):
        self.node_id = node_id
        self.shard_id = shard_id
        self.host = host
        self.port = port
        self.chain: List[Block] = []
        self.pending_transactions: List[Dict] = []
        self.peers: Dict[str, tuple] = {}  # node_id -> (host, port)
        self.is_running = False
        self.is_malicious = is_malicious
        self.reputation_system = reputation_system
        self.last_block_time = time.time()
        self.voting_history: List[Dict] = []
        self.stop_event = asyncio.Event()
        self.start_time = time.time()
        self._create_genesis_block()

    def _create_genesis_block(self):
        """Create the genesis block"""
        genesis_block = Block(
            index=0,
            timestamp=time.time(),
            transactions=[],
            previous_hash="0" * 64,
            shard_id=self.shard_id
        )
        self.chain.append(genesis_block)

    async def start(self):
        """Start the node"""
        self.is_running = True
        logger.info(f"Node {self.node_id} started on {self.host}:{self.port} (Shard {self.shard_id})")

    async def stop(self):
        """Stop the node"""
        self.stop_event.set()
        self.is_running = False
        logger.info(f"Node {self.node_id} stopped")

    async def add_peer(self, node_id: str, host: str, port: int):
        """Add a peer to the node's peer list"""
        if node_id != self.node_id:
            self.peers[node_id] = (host, port)
            logger.info(f"Node {self.node_id} added peer {node_id}")

    def _simulate_malicious_behavior(self) -> Dict:
        """Simulate malicious behavior for malicious nodes with various attack patterns"""
        if not self.is_malicious:
            return {
                "block_delay": np.random.normal(2.0, 0.5),
                "voting_mismatch": random.uniform(0.0, 0.1),
                "penalties": np.random.poisson(0.1),
                "is_attack_phase": False,
                "attack_type": "honest"
            }
            
        current_time = time.time()
        
        # Attack pattern selection based on node ID
        if self.node_id == "node_1":
            # On-off attack pattern
            cycle_duration = 30
            attack_duration = 10
            cycle_position = current_time % cycle_duration
            is_attack_phase = cycle_position < attack_duration
            
            if is_attack_phase:
                return {
                    "block_delay": np.random.normal(15.0, 2.0),
                    "voting_mismatch": random.uniform(0.7, 0.9),
                    "penalties": np.random.poisson(2),
                    "is_attack_phase": True,
                    "attack_type": "on_off"
                }
                
        elif self.node_id == "node_2":
            # Gradual degradation attack
            degradation_factor = min(1.0, (current_time - self.start_time) / 300)  # 5 minutes
            return {
                "block_delay": np.random.normal(2.0 + degradation_factor * 10, 0.5),
                "voting_mismatch": random.uniform(0.1, 0.1 + degradation_factor * 0.8),
                "penalties": np.random.poisson(0.1 + degradation_factor),
                "is_attack_phase": True,
                "attack_type": "gradual"
            }
            
        elif self.node_id == "node_3":
            # Random burst attack
            if random.random() < 0.1:  # 10% chance of burst
                return {
                    "block_delay": np.random.normal(20.0, 5.0),
                    "voting_mismatch": random.uniform(0.8, 1.0),
                    "penalties": np.random.poisson(5),
                    "is_attack_phase": True,
                    "attack_type": "burst"
                }
                
        elif self.node_id == "node_4":
            # Adaptive attack - changes behavior based on reputation
            current_rep = self.reputation_system.get_reputation(self.node_id)
            if current_rep > 0.7:
                return {
                    "block_delay": np.random.normal(20.0, 5.0),
                    "voting_mismatch": random.uniform(0.8, 1.0),
                    "penalties": np.random.poisson(5),
                    "is_attack_phase": True,
                    "attack_type": "adaptive"
                }
            else:
                return {
                    "block_delay": np.random.normal(2.0, 0.5),
                    "voting_mismatch": random.uniform(0.0, 0.1),
                    "penalties": np.random.poisson(0.1),
                    "is_attack_phase": False,
                    "attack_type": "adaptive"
                }
                
        # Default honest behavior
        return {
            "block_delay": np.random.normal(2.0, 0.5),
            "voting_mismatch": random.uniform(0.0, 0.1),
            "penalties": np.random.poisson(0.1),
            "is_attack_phase": False,
            "attack_type": "honest"
        }

    async def add_transaction(self, transaction: Dict):
        """Add a transaction to the pending transactions"""
        if transaction.get("shard_id") == self.shard_id:
            # Simulate malicious behavior if node is malicious
            behavior = self._simulate_malicious_behavior()
            
            # Add artificial delay for malicious nodes
            if self.is_malicious and behavior["block_delay"] > 3:
                await asyncio.sleep(behavior["block_delay"])
                
            # Simulate voting mismatch
            if self.is_malicious and behavior["voting_mismatch"] > 0.2:
                transaction["candidate_id"] = f"candidate_{random.randint(1, 3)}"
                
            self.pending_transactions.append(transaction)
            logger.info(f"Node {self.node_id} added transaction: {transaction}")

    async def mine_block(self) -> Optional[Block]:
        """Mine a new block with pending transactions"""
        if not self.pending_transactions:
            return None

        # Simulate malicious behavior
        behavior = self._simulate_malicious_behavior()
        
        # Add artificial delay for malicious nodes
        if self.is_malicious and behavior["block_delay"] > 3:
            await asyncio.sleep(behavior["block_delay"])

        previous_block = self.chain[-1]
        new_block = Block(
            index=previous_block.index + 1,
            timestamp=time.time(),
            transactions=self.pending_transactions.copy(),
            previous_hash=previous_block.hash,
            shard_id=self.shard_id
        )

        self.chain.append(new_block)
        self.pending_transactions = []
        logger.info(f"Node {self.node_id} mined block {new_block.index}")
        return new_block

    def is_chain_valid(self) -> bool:
        """Validate the blockchain"""
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i-1]

            if current.previous_hash != previous.hash:
                return False
            if current.hash != current.calculate_hash():
                return False
        return True

    def get_chain_state(self) -> Dict:
        """Get the current state of the blockchain"""
        return {
            "node_id": self.node_id,
            "shard_id": self.shard_id,
            "chain_length": len(self.chain),
            "pending_transactions": len(self.pending_transactions),
            "peers": len(self.peers),
            "is_malicious": self.is_malicious
        }

    def get_current_behavior(self) -> Dict:
        """Get current behavior metrics"""
        behavior = self._simulate_malicious_behavior()
        behavior["node_id"] = self.node_id
        return behavior
        
    async def start_mining(self):
        """Start the mining process"""
        self.is_running = True
        while not self.stop_event.is_set():
            try:
                # Mine a new block
                if self.pending_transactions:
                    behavior = self._simulate_malicious_behavior()
                    await asyncio.sleep(behavior["block_delay"])
                    await self.mine_block()
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Mining error in node {self.node_id}: {str(e)}")
                await asyncio.sleep(1) 