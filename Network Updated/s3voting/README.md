# Sharded Blockchain Voting System Simulation

This is a simulation of a sharded blockchain-based voting system. The system demonstrates how a blockchain network can be divided into shards to improve scalability while maintaining security and decentralization.

## Project Structure

```
s3voting/
├── simulation/
│   ├── blockchain/
│   │   ├── node.py      # Blockchain node implementation
│   │   └── shard.py     # Shard management
│   └── run_simulation.py # Main simulation script
├── requirements.txt
└── README.md
```

## Features

- Sharded blockchain architecture
- Peer-to-peer node communication within shards
- Distributed voting system
- Transaction processing and block mining
- Chain validation and consensus

## Requirements

- Python 3.7+
- Required packages listed in `requirements.txt`

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Simulation

To run the simulation:

```bash
python simulation/run_simulation.py
```

The simulation will:
1. Create multiple nodes and assign them to shards
2. Connect nodes within the same shard
3. Simulate voting transactions
4. Process transactions and mine blocks
5. Display final state and statistics

## Configuration

You can modify the following parameters in `run_simulation.py`:
- `num_nodes`: Number of nodes in the network (default: 10)
- `num_shards`: Number of shards (default: 5)
- `num_voters`: Number of voters in the simulation (default: 30)

## Output

The simulation provides detailed logging of:
- Node creation and shard assignment
- Peer connections
- Vote casting
- Block mining
- Final blockchain state and statistics 