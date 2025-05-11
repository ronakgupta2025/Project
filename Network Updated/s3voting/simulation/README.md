# Blockchain E-Voting Simulation

This is a Streamlit-based web application for simulating and analyzing blockchain-based e-voting systems with various attack patterns and configurations.

## Features

- Multiple attack pattern simulations (On-Off, Persistent, Sporadic, Targeted)
- Configurable number of shards and nodes
- Real-time progress monitoring
- Interactive visualizations
- Comparative analysis of multiple simulation runs
- CSV export of simulation results

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

## Usage

1. Configure simulation parameters in the sidebar:
   - Total number of nodes
   - Malicious node ratio
   - Number of shards
   - Number of voters
   - Attack pattern
   - Shard reshuffling option

2. Click "Run Simulation" to start a new simulation

3. View results:
   - Processing time vs. shard count
   - Throughput comparison
   - Reputation trajectories
   - Detailed metrics table

4. Compare multiple runs:
   - Select runs to compare from the dropdown
   - View comparative visualizations
   - Export results to CSV

## Directory Structure

```
simulation/
├── app.py              # Streamlit web application
├── run_simulation.py   # Core simulation logic
├── blockchain/         # Blockchain implementation
│   ├── node.py        # Node implementation
│   ├── shard.py       # Shard management
│   └── reputation.py  # Reputation system
├── utils/             # Utility functions
├── voting/            # Voting logic
├── simulation_results/ # Directory for simulation results
└── requirements.txt   # Python dependencies
```

## Results

Simulation results are saved in the `simulation_results/` directory with timestamps and parameter information. Each run generates:
- Processing time metrics
- Throughput measurements
- Reputation scores
- Malicious node distribution
- Shard-level statistics

## Contributing

Feel free to submit issues and enhancement requests! 