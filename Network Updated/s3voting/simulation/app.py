import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import asyncio
import json
from typing import Dict, List, Optional
import time

from run_simulation import Simulation

# Constants
RESULTS_DIR = "simulation_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Session state initialization
if 'runs' not in st.session_state:
    st.session_state.runs = []
if 'current_run' not in st.session_state:
    st.session_state.current_run = None
if 'simulation_status' not in st.session_state:
    st.session_state.simulation_status = {
        'current_phase': '',
        'progress': 0,
        'current_vote': 0,
        'total_votes': 0,
        'current_shard': 0,
        'malicious_nodes': [],
        'reputation_scores': {},
        'last_update': time.time()
    }

def update_simulation_status(sim: Simulation, current_vote: int, total_votes: int):
    """Update simulation status for UI feedback"""
    st.session_state.simulation_status.update({
        'current_phase': 'Voting',
        'progress': (current_vote / total_votes) * 100,
        'current_vote': current_vote,
        'total_votes': total_votes,
        'current_shard': sim.shard_manager.num_shards,
        'malicious_nodes': sim.malicious_nodes,
        'reputation_scores': {node.node_id: sim.reputation_system.get_reputation(node.node_id) 
                            for node in sim.nodes},
        'last_update': time.time()
    })

def save_run_results(params: Dict, metrics: Dict):
    """Save simulation results to CSV"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{RESULTS_DIR}/simulation_{timestamp}.csv"
    
    # Flatten nested dictionaries in metrics
    flattened_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            # Handle nested dictionaries (like reputation_history)
            for subkey, subvalue in value.items():
                flattened_metrics[f"{key}_{subkey}"] = subvalue
        else:
            flattened_metrics[key] = value
    
    # Convert metrics to DataFrame
    df = pd.DataFrame([flattened_metrics])
    df['timestamp'] = timestamp
    
    # Add simulation parameters
    for key, value in params.items():
        df[key] = value
    
    df.to_csv(filename, index=False)
    return filename

def flatten_metrics(metrics: Dict) -> Dict:
    """Helper function to flatten nested dictionaries in metrics"""
    flattened = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                if isinstance(subvalue, (list, tuple)):
                    # Handle lists/tuples by taking the last value
                    flattened[f"{key}_{subkey}"] = subvalue[-1] if subvalue else 0
                else:
                    flattened[f"{key}_{subkey}"] = subvalue
        elif isinstance(value, (list, tuple)):
            # Handle lists/tuples by taking the last value
            flattened[key] = value[-1] if value else 0
        else:
            flattened[key] = value
    return flattened

def plot_processing_time(runs: List[Dict]):
    """Plot processing time vs shard count"""
    fig = go.Figure()
    
    for run in runs:
        try:
            # Flatten metrics before creating DataFrame
            flattened_metrics = flatten_metrics(run['metrics'])
            df = pd.DataFrame([flattened_metrics])
            
            # Ensure numeric values
            if 'num_shards' in df.columns and 'processing_time' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['num_shards'].astype(float),
                    y=df['processing_time'].astype(float),
                    name=f"Run {run['timestamp']}",
                    mode='lines+markers'
                ))
        except Exception as e:
            st.warning(f"Could not plot data for run {run['timestamp']}: {str(e)}")
    
    fig.update_layout(
        title="Processing Time vs Shard Count",
        xaxis_title="Number of Shards",
        yaxis_title="Processing Time (s)",
        showlegend=True
    )
    
    return fig

def plot_throughput_comparison(runs: List[Dict]):
    """Plot throughput comparison with/without reshuffling"""
    fig = go.Figure()
    
    for run in runs:
        try:
            # Flatten metrics before creating DataFrame
            flattened_metrics = flatten_metrics(run['metrics'])
            df = pd.DataFrame([flattened_metrics])
            
            # Ensure numeric values and calculate throughput
            if 'voters' in df.columns and 'processing_time' in df.columns:
                voters = float(df['voters'].iloc[0])
                proc_time = float(df['processing_time'].iloc[0])
                throughput = voters / proc_time if proc_time > 0 else 0
                
                label = "With Reshuffling" if run['params']['enable_reshuffling'] else "Without Reshuffling"
                fig.add_trace(go.Scatter(
                    x=[voters],
                    y=[throughput],
                    name=f"{label} - Run {run['timestamp']}",
                    mode='lines+markers'
                ))
        except Exception as e:
            st.warning(f"Could not plot data for run {run['timestamp']}: {str(e)}")
    
    fig.update_layout(
        title="Throughput Comparison",
        xaxis_title="Number of Voters",
        yaxis_title="Votes per Second",
        showlegend=True
    )
    
    return fig

def plot_reputation_trajectories(runs: List[Dict]):
    """Plot reputation trajectories for malicious nodes"""
    fig = go.Figure()
    
    for run in runs:
        try:
            # Flatten metrics before creating DataFrame
            flattened_metrics = flatten_metrics(run['metrics'])
            df = pd.DataFrame([flattened_metrics])
            
            # Filter reputation columns for malicious nodes
            rep_columns = [col for col in df.columns if col.startswith('reputation_history_') and 'malicious' in col]
            for col in rep_columns:
                node_id = col.replace('reputation_history_', '')
                if 'voters' in df.columns and col in df.columns:
                    fig.add_trace(go.Scatter(
                        x=[float(df['voters'].iloc[0])],
                        y=[float(df[col].iloc[0])],
                        name=f"{node_id} - Run {run['timestamp']}",
                        mode='lines+markers'
                    ))
        except Exception as e:
            st.warning(f"Could not plot data for run {run['timestamp']}: {str(e)}")
    
    fig.update_layout(
        title="Reputation Trajectories for Malicious Nodes",
        xaxis_title="Number of Voters",
        yaxis_title="Reputation Score",
        showlegend=True
    )
    
    return fig

def calculate_voting_results(metrics: Dict) -> Dict:
    """Calculate voting results from metrics"""
    results = {
        'total_votes': 0,
        'by_candidate': {},
        'by_shard': {},
        'by_shard_candidate': {}
    }
    
    # Extract voting data from metrics
    if 'voting_results' in metrics:
        voting_data = metrics['voting_results']
        
        # Calculate total votes
        results['total_votes'] = len(voting_data)
        
        # Calculate votes by candidate
        for vote in voting_data:
            candidate = vote.get('candidate_id', 'unknown')
            shard = vote.get('shard_id', 'unknown')
            
            # Update candidate totals
            results['by_candidate'][candidate] = results['by_candidate'].get(candidate, 0) + 1
            
            # Update shard totals
            if shard not in results['by_shard']:
                results['by_shard'][shard] = 0
            results['by_shard'][shard] += 1
            
            # Update shard-candidate totals
            if shard not in results['by_shard_candidate']:
                results['by_shard_candidate'][shard] = {}
            results['by_shard_candidate'][shard][candidate] = results['by_shard_candidate'][shard].get(candidate, 0) + 1
    
    return results

def show_voting_results_text(runs: List[Dict]):
    """Show voting results as a text table (not a graph)"""
    if not runs:
        st.info("No voting results to display.")
        return
    latest_run = runs[-1]
    metrics = latest_run['metrics']
    if 'voting_results' not in metrics:
        st.info("No voting results found in metrics.")
        return
    voting_data = metrics['voting_results']
    if not voting_data:
        st.info("No votes were cast.")
        return
    # Build table: Shard x Candidate
    shard_candidate = {}
    candidate_totals = {}
    for vote in voting_data:
        shard = vote.get('shard_id', 'unknown')
        candidate = vote.get('candidate_id', 'unknown')
        if shard not in shard_candidate:
            shard_candidate[shard] = {}
        if candidate not in shard_candidate[shard]:
            shard_candidate[shard][candidate] = 0
        shard_candidate[shard][candidate] += 1
        candidate_totals[candidate] = candidate_totals.get(candidate, 0) + 1
    # Prepare DataFrame
    df = pd.DataFrame(shard_candidate).fillna(0).astype(int).T
    st.subheader("Votes per Candidate in Each Shard")
    st.dataframe(df)
    st.subheader("Total Votes per Candidate")
    total_df = pd.DataFrame.from_dict(candidate_totals, orient='index', columns=['Total Votes'])
    st.dataframe(total_df)

def plot_voters_vs_time(runs: List[Dict]):
    """Plot number of voters processed vs. cumulative time, with a line for each shard count (K), using granular data points."""
    if not runs:
        return None
    fig = go.Figure()
    for run in runs:
        metrics = run['metrics']
        params = run['params']
        K = params.get('num_shards', 'K?')
        voters_vs_time = metrics.get('voters_vs_time', [])
        if not voters_vs_time:
            continue
        voters, times = zip(*voters_vs_time)
        # Convert time to hours
        times = [t / 3600.0 for t in times]
        fig.add_trace(go.Scatter(
            x=voters,
            y=times,
            mode='lines+markers',
            name=f'K={K}'
        ))
    fig.update_layout(
        title="Number of Voters Processed vs. Time (by Shard Count)",
        xaxis_title="Number of voters",
        yaxis_title="Time (h)",
        xaxis_type='log',
        legend_title="Shards (K)",
        showlegend=True
    )
    return fig

def display_simulation_status():
    """Display current simulation status"""
    status = st.session_state.simulation_status
    
    # Create columns for status display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Phase", status['current_phase'])
        st.metric("Progress", f"{status['progress']:.1f}%")
    
    with col2:
        st.metric("Current Vote", f"{status['current_vote']}/{status['total_votes']}")
        st.metric("Number of Shards", status['current_shard'])
    
    with col3:
        st.metric("Malicious Nodes", len(status['malicious_nodes']))
        st.metric("Last Update", f"{time.time() - status['last_update']:.1f}s ago")
    
    # Display reputation scores
    if status['reputation_scores']:
        st.subheader("Node Reputation Scores")
        rep_df = pd.DataFrame.from_dict(status['reputation_scores'], orient='index', columns=['Reputation'])
        st.dataframe(rep_df)

def main():
    st.title("Blockchain E-Voting Simulation")
    
    # Sidebar controls
    st.sidebar.header("Simulation Parameters")
    
    # Basic parameters
    num_nodes = st.sidebar.number_input("Total Nodes", min_value=10, max_value=100, value=10)
    malicious_ratio = st.sidebar.slider("Malicious Node Ratio", 0.0, 0.5, 0.2, 0.05)
    num_shards = st.sidebar.number_input("Number of Shards", min_value=2, max_value=10, value=5)
    num_voters = st.sidebar.number_input("Number of Voters", min_value=10, max_value=500, value=10)
    
    # Attack pattern selection
    attack_pattern = st.sidebar.selectbox(
        "Attack Pattern",
        ["On-Off", "Persistent", "Sporadic", "Targeted"]
    )
    
    # Additional options
    enable_reshuffling = st.sidebar.checkbox("Enable Shard Reshuffling", value=True)
    
    # Run simulation button
    if st.sidebar.button("Run Simulation"):
        with st.spinner("Running simulation..."):
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Initialize simulation
                sim = Simulation(
                    num_nodes=num_nodes,
                    num_shards=num_shards,
                    malicious_ratio=malicious_ratio,
                    attack_pattern=attack_pattern,
                    enable_reshuffling=enable_reshuffling,
                    num_voters=num_voters
                )
                
                # Run simulation
                params = {
                    'num_nodes': num_nodes,
                    'malicious_ratio': malicious_ratio,
                    'num_shards': num_shards,
                    'num_voters': num_voters,
                    'attack_pattern': attack_pattern,
                    'enable_reshuffling': enable_reshuffling
                }
                
                # Create a placeholder for real-time status
                status_placeholder = st.empty()
                
                # Run simulation and collect metrics
                metrics = asyncio.run(sim.run())
                
                # Save results
                filename = save_run_results(params, metrics)
                
                # Add to session state
                st.session_state.runs.append({
                    'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                    'params': params,
                    'metrics': metrics,
                    'filename': filename
                })
                
                st.success(f"Simulation completed! Results saved to {filename}")
                
            except Exception as e:
                st.error(f"An error occurred during simulation: {str(e)}")
                st.error("Please check the parameters and try again.")
    
    # Display current simulation status
    if st.session_state.simulation_status['current_phase']:
        st.header("Current Simulation Status")
        display_simulation_status()
    
    # Display results
    if st.session_state.runs:
        st.header("Simulation Results")
        
        # Select runs to compare
        selected_runs = st.multiselect(
            "Select runs to compare",
            options=[run['timestamp'] for run in st.session_state.runs],
            default=[run['timestamp'] for run in st.session_state.runs[-2:]]
        )
        
        if selected_runs:
            filtered_runs = [run for run in st.session_state.runs if run['timestamp'] in selected_runs]
            
            try:
                # Create tabs for different visualizations
                tab1, tab2, tab3, tab4 = st.tabs([
                    "Processing Time", 
                    "Throughput", 
                    "Reputation",
                    "Voting Results"
                ])
                
                with tab1:
                    st.plotly_chart(plot_processing_time(filtered_runs))
                    st.plotly_chart(plot_voters_vs_time(filtered_runs))
                
                with tab2:
                    st.plotly_chart(plot_throughput_comparison(filtered_runs))
                
                with tab3:
                    st.plotly_chart(plot_reputation_trajectories(filtered_runs))
                
                with tab4:
                    show_voting_results_text(filtered_runs)
                
                # Display metrics table
                st.subheader("Detailed Metrics")
                for run in filtered_runs:
                    st.write(f"Run {run['timestamp']}")
                    try:
                        flattened_metrics = flatten_metrics(run['metrics'])
                        st.dataframe(pd.DataFrame([flattened_metrics]))
                    except Exception as e:
                        st.warning(f"Could not display metrics for run {run['timestamp']}: {str(e)}")
                    
            except Exception as e:
                st.error(f"An error occurred while displaying results: {str(e)}")
                st.error("Please try selecting different runs or parameters.")

if __name__ == "__main__":
    main() 