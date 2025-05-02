# atl/verification/main_verification.py

from .synthetic_verification import verify_synthetic_experiment
from .real_data_verification import verify_real_data_experiment
from .early_stopping_verification import verify_early_stopping
import numpy as np
import pandas as pd

def run_verification():
    """Run all verification experiments."""
    print("Starting verification against paper results...")
    
    # Verify synthetic experiment
    print("\n=== Synthetic Experiment Verification ===")
    synthetic_results = verify_synthetic_experiment()
    
    # Verify real data experiments
    print("\n=== Real Data Experiment Verification ===")
    datasets = ["mnist", "fashion_mnist", "cifar10"]
    real_data_results = {}
    
    for dataset in datasets:
        print(f"\nVerifying {dataset}...")
        real_data_results[dataset] = verify_real_data_experiment(
            dataset, n_rounds=20, n_samples_per_round=500, n_test_per_round=100, n_runs=3
        )
    
    # Verify early stopping
    print("\n=== Early Stopping Verification ===")
    early_stopping_results = {}
    
    for dataset in datasets:
        print(f"\nVerifying early stopping on {dataset}...")
        early_stopping_results[dataset] = verify_early_stopping(
            dataset, n_rounds=40, n_samples_per_round=500, n_test_per_round=100
        )
    
    # Summary of verification
    print("\n=== Verification Summary ===")
    
    # Compare estimation errors with paper Table 1
    print("\nEstimation Error Comparison with Paper Table 1:")
    for dataset in datasets:
        print(f"\n{dataset} Results:")
        for method in ["ARE", "AT", "ASE", "ATL-NF"]:
            paper_values = get_paper_table1_values(dataset, method)
            our_values = get_our_estimation_errors(real_data_results[dataset], method)
            
            print(f"  {method}:")
            print(f"    Paper: {paper_values}")
            print(f"    Ours:  {our_values}")
    
    # Compare holdout risks with paper Table 2
    print("\nHoldout Risk Comparison with Paper Table 2:")
    for dataset in datasets:
        print(f"\n{dataset} Results:")
        for method in ["ATL-NF", "ATL-RF", "ATL"]:
            paper_values = get_paper_table2_values(dataset, method)
            our_values = get_our_holdout_risks(real_data_results[dataset], method)
            
            print(f"  {method}:")
            print(f"    Paper: {paper_values}")
            print(f"    Ours:  {our_values}")
    
    # Compare estimation errors with feedback with paper Table 3
    print("\nEstimation Error with Feedback Comparison with Paper Table 3:")
    for dataset in datasets:
        print(f"\n{dataset} Results:")
        for method in ["ATL-RF", "ATL"]:
            paper_values = get_paper_table3_values(dataset, method)
            our_values = get_our_estimation_errors(real_data_results[dataset], method)
            
            print(f"  {method}:")
            print(f"    Paper: {paper_values}")
            print(f"    Ours:  {our_values}")
    
    # Early stopping effectiveness
    print("\nEarly Stopping Effectiveness:")
    for dataset in datasets:
        results = early_stopping_results[dataset]
        
        with_stop = results['with_stopping']
        without_stop = results['without_stopping']
        
        print(f"\n{dataset} Results:")
        print(f"  With early stopping:")
        print(f"    Rounds used: {with_stop['rounds_used']} / {without_stop['rounds_used']}")
        print(f"    Labels used: {with_stop['labels_used']} / {without_stop['labels_used']}")
        print(f"    Holdout risk: {with_stop['holdout_risk']:.4f} vs {without_stop['holdout_risk']:.4f}")
        print(f"    Label savings: {without_stop['labels_used'] - with_stop['labels_used']} labels ({(without_stop['labels_used'] - with_stop['labels_used']) / without_stop['labels_used'] * 100:.1f}%)")
    
    return {
        'synthetic_results': synthetic_results,
        'real_data_results': real_data_results,
        'early_stopping_results': early_stopping_results
    }

# Helper functions to get paper values (these would be hardcoded from the paper)
def get_paper_table1_values(dataset, method):
    # These values should be extracted from Table 1 in the paper
    # Format: [round 4, round 8, round 12, round 16, round 20]
    
    table1_values = {
        "mnist": {
            "ARE": ["5.27 ± 5.42", "6.39 ± 1.54", "2.96 ± 3.45", "8.85 ± 4.31", "8.31 ± 3.96"],
            "AT": ["16.3 ± 24.5", "32.8 ± 22.1", "6.93 ± 18.0", "8.72 ± 3.59", "3.11 ± 2.98"],
            "ASE": ["3.45 ± 2.76", "1.45 ± 1.00", "2.17 ± 5.06", "4.00 ± 2.37", "5.88 ± 5.27"],
            "ATL-NF": ["2.57 ± 1.17", "0.79 ± 1.15", "0.17 ± 0.15", "0.56 ± 0.30", "1.32 ± 0.37"]
        },
        "fashion_mnist": {
            "ARE": ["4.24 ± 3.01", "4.62 ± 7.77", "8.63 ± 2.47", "5.71 ± 1.87", "23.78 ± 1.75"],
            "AT": ["11.9 ± 6.1", "36.1 ± 30.7", "34.1 ± 31.4", "28.0 ± 36.9", "22.5 ± 25.7"],
            "ASE": ["11.1 ± 3.63", "3.72 ± 3.53", "3.56 ± 8.78", "5.29 ± 9.78", "8.42 ± 6.72"],
            "ATL-NF": ["3.64 ± 1.61", "0.67 ± 0.38", "0.96 ± 0.16", "0.98 ± 0.43", "3.04 ± 1.37"]
        },
        "cifar10": {
            "ARE": ["10.1 ± 8.79", "13.8 ± 13.0", "22.2 ± 14.7", "21.9 ± 31.4", "14.1 ± 13.4"],
            "AT": ["6.89 ± 6.98", "12.0 ± 7.18", "21.8 ± 5.73", "12.9 ± 9.76", "38.9 ± 25.6"],
            "ASE": ["10.9 ± 3.67", "6.51 ± 2.87", "7.53 ± 1.46", "17.6 ± 2.66", "23.2 ± 6.10"],
            "ATL-NF": ["8.83 ± 7.79", "3.06 ± 5.04", "4.95 ± 7.12", "7.94 ± 5.22", "6.20 ± 5.79"]
        }
    }
    
    return table1_values.get(dataset, {}).get(method, ["N/A"] * 5)

def get_paper_table2_values(dataset, method):
    # These values should be extracted from Table 2 in the paper
    # Format: [round 4, round 8, round 12, round 16, round 20]
    
    table2_values = {
        "mnist": {
            "ATL-NF": ["0.92 ± 0.06", "0.55 ± 0.08", "0.46 ± 0.06", "0.32 ± 0.04", "0.22 ± 0.02"],
            "ATL-RF": ["0.92 ± 0.12", "0.54 ± 0.02", "0.41 ± 0.05", "0.29 ± 0.03", "0.21 ± 0.02"],
            "ATL": ["0.88 ± 0.07", "0.53 ± 0.04", "0.39 ± 0.03", "0.26 ± 0.01", "0.19 ± 0.03"]
        },
        "fashion_mnist": {
            "ATL-NF": ["0.75 ± 0.03", "0.69 ± 0.02", "0.61 ± 0.02", "0.57 ± 0.04", "0.56 ± 0.03"],
            "ATL-RF": ["0.75 ± 0.04", "0.68 ± 0.02", "0.61 ± 0.01", "0.58 ± 0.06", "0.56 ± 0.04"],
            "ATL": ["0.74 ± 0.03", "0.65 ± 0.04", "0.59 ± 0.02", "0.56 ± 0.03", "0.51 ± 0.01"]
        },
        "cifar10": {
            "ATL-NF": ["1.91 ± 0.04", "1.76 ± 0.05", "1.72 ± 0.01", "1.66 ± 0.02", "1.55 ± 0.03"],
            "ATL-RF": ["1.91 ± 0.03", "1.77 ± 0.04", "1.69 ± 0.03", "1.60 ± 0.04", "1.54 ± 0.07"],
            "ATL": ["1.90 ± 0.05", "1.76 ± 0.02", "1.65 ± 0.03", "1.58 ± 0.02", "1.53 ± 0.02"]
        }
    }
    
    return table2_values.get(dataset, {}).get(method, ["N/A"] * 5)

def get_paper_table3_values(dataset, method):
    # These values should be extracted from Table 3 in the paper
    # Format: [round 4, round 8, round 12, round 16, round 20]
    
    table3_values = {
        "mnist": {
            "ATL-RF": ["26.8 ± 21.4", "21.4 ± 17.0", "3.54 ± 4.01", "5.54 ± 3.21", "7.62 ± 4.41"],
            "ATL": ["14.6 ± 22.1", "16.9 ± 13.7", "3.19 ± 2.63", "4.15 ± 3.20", "1.87 ± 1.41"]
        },
        "fashion_mnist": {
            "ATL-RF": ["10.2 ± 9.30", "4.41 ± 3.77", "2.19 ± 5.53", "5.69 ± 4.52", "11.6 ± 7.51"],
            "ATL": ["2.50 ± 2.93", "1.94 ± 2.25", "1.78 ± 1.07", "6.32 ± 5.41", "5.03 ± 4.41"]
        },
        "cifar10": {
            "ATL-RF": ["20.6 ± 17.6", "19.1 ± 13.7", "9.82 ± 8.03", "33.6 ± 30.5", "24.8 ± 32.4"],
            "ATL": ["11.6 ± 13.4", "5.11 ± 3.45", "8.81 ± 6.51", "11.9 ± 16.7", "6.57 ± 6.29"]
        }
    }
    
    return table3_values.get(dataset, {}).get(method, ["N/A"] * 5)

def get_our_estimation_errors(results, method):
    # Format our results similar to the paper for comparison
    rounds_to_check = [4, 8, 12, 16, 20]
    formatted_results = []
    
    for round_idx in rounds_to_check:
        if round_idx - 1 < len(results['mean_errors'][method]):
            value = results['mean_errors'][method][round_idx - 1] * 1000  # Convert to x10^-3
            std = results['std_errors'][method][round_idx - 1] * 1000
            formatted_results.append(f"{value:.2f} ± {std:.2f}")
        else:
            formatted_results.append("N/A")
    
    return formatted_results

def get_our_holdout_risks(results, method):
    # Format our holdout risks similar to the paper for comparison
    rounds_to_check = [4, 8, 12, 16, 20]
    formatted_results = []
    
    for round_idx in rounds_to_check:
        if method in results['holdout_risks'] and len(results['holdout_risks'][method]) > 0:
            # Use true risk as a proxy for holdout risk at specific rounds
            risks = [run[round_idx - 1] if round_idx - 1 < len(run) else None 
                    for run in [[r['true_risk_history'] for r in results['holdout_risks'][method]]]]
            risks = [r for r in risks if r is not None]
            
            if risks:
                value = np.mean(risks)
                std = np.std(risks)
                formatted_results.append(f"{value:.2f} ± {std:.2f}")
            else:
                formatted_results.append("N/A")
        else:
            formatted_results.append("N/A")
    
    return formatted_results

if __name__ == "__main__":
    verification_results = run_verification()