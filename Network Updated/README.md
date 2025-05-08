# Blockchain Voting Simulation

This project simulates a blockchain-based voting system with reputation-based security mechanisms. The simulation includes various attack patterns and demonstrates how the system detects and mitigates malicious behavior.

## Features

- Multi-shard blockchain architecture
- Reputation-based security system
- Multiple attack pattern simulations
- Real-time monitoring and visualization
- Asynchronous node operations

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Simulation

1. Start the simulation:
```bash
python s3voting/simulation/run_simulation.py
```

The simulation will:
- Create 10 nodes (4 malicious, 6 honest)
- Distribute nodes across 5 shards
- Process 100 voting transactions
- Run for approximately 60 seconds
- Generate visualization of attack patterns
- Save behavior metrics

## Output

The simulation generates several outputs:
- Real-time logs showing:
  - Voting progress
  - Mining operations
  - Reputation updates
  - Node states
- `simulation_results/attack_patterns.png`: Visualization of attack patterns
- `simulation_results/behavior_metrics.txt`: Detailed behavior metrics

## Configuration

You can modify the simulation parameters in `run_simulation.py`:
- `num_nodes`: Total number of nodes (default: 10)
- `num_shards`: Number of shards (default: 5)
- `malicious_ratio`: Ratio of malicious nodes (default: 0.2)

## Project Structure

```
s3voting/
├── simulation/
│   ├── run_simulation.py    # Main simulation script
│   └── ...
├── blockchain/
│   ├── node.py             # Blockchain node implementation
│   ├── shard.py            # Shard management
│   └── reputation.py       # Reputation system
└── ...
```

## Monitoring

During the simulation, you can monitor:
- Voting progress
- Mining operations
- Reputation scores
- Node states
- Attack patterns
- Trust levels

## Troubleshooting

If you encounter any issues:
1. Ensure all dependencies are installed correctly
2. Check Python version (3.8+ required)
3. Verify virtual environment is activated
4. Check log output for specific error messages

## License

[Your License Here] 