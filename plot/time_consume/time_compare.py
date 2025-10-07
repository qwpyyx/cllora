import numpy as np
import matplotlib.pyplot as plt
import os
import re
from typing import Tuple, Dict, List

# Use Matplotlib's built-in font (DejaVu Sans) to avoid "font not found" errors
# No need for extra font config—this works cross-platform by default
plt.rcParams["axes.unicode_minus"] = False  # Fix negative sign display


def load_time_data(adaptive_path: str, baseline_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load adaptive and baseline time data for a single client"""
    try:
        adaptive_times = np.load(adaptive_path, allow_pickle=True)
        baseline_times = np.load(baseline_path, allow_pickle=True)
        print(f"Successfully loaded data:")
        print(f"  Adaptive: {len(adaptive_times)} batches | Path: {adaptive_path}")
        print(f"  Baseline: {len(baseline_times)} batches | Path: {baseline_path}")
        return adaptive_times, baseline_times
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        raise
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def calculate_statistics(times: np.ndarray) -> Dict[str, float]:
    """Calculate basic statistics for time data"""
    return {
        "Mean (ms)": np.mean(times),
        "Std (ms)": np.std(times),
        "Median (ms)": np.median(times),
        "Min (ms)": np.min(times),
        "Max (ms)": np.max(times),
        "Total Samples": len(times)
    }


def analyze_time_difference(adaptive_times: np.ndarray, baseline_times: np.ndarray) -> Dict[str, float]:
    """Analyze time difference between adaptive and baseline modes"""
    min_length = min(len(adaptive_times), len(baseline_times))
    adaptive_truncated = adaptive_times[:min_length]
    baseline_truncated = baseline_times[:min_length]
    extra_time = adaptive_truncated - baseline_truncated

    return {
        "Avg Extra Time (ms)": np.mean(extra_time),
        "Extra Time Std (ms)": np.std(extra_time),
        "Total Extra Time (ms)": np.sum(extra_time),
        "Avg Extra Time (%)": np.mean(extra_time / baseline_truncated) * 100
    }


def plot_comparison(adaptive_times: np.ndarray, baseline_times: np.ndarray, task_id: int, cid: int,
                    save_dir: str = None):
    """Generate comparison plots for a single client"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Task {task_id} - Client {cid} - Adaptive vs Baseline', fontsize=16)

    # 1. Box Plot: Distribution Comparison
    axes[0, 0].boxplot([baseline_times, adaptive_times], labels=['Baseline', 'Adaptive'])
    axes[0, 0].set_title('Batch Time Distribution (Box Plot)')
    axes[0, 0].set_ylabel('Time (ms)')
    axes[0, 0].grid(alpha=0.3)

    # 2. Histogram: Frequency Distribution
    bins = np.linspace(
        min(min(baseline_times), min(adaptive_times)),
        max(max(baseline_times), max(adaptive_times)),
        30
    )
    axes[0, 1].hist(baseline_times, bins=bins, alpha=0.5, label='Baseline')
    axes[0, 1].hist(adaptive_times, bins=bins, alpha=0.5, label='Adaptive')
    axes[0, 1].set_title('Time Frequency Distribution (Histogram)')
    axes[0, 1].set_xlabel('Time (ms)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # 3. Cumulative Distribution Plot
    axes[1, 0].hist(baseline_times, bins=bins, cumulative=True, density=True, alpha=0.5, label='Baseline')
    axes[1, 0].hist(adaptive_times, bins=bins, cumulative=True, density=True, alpha=0.5, label='Adaptive')
    axes[1, 0].set_title('Cumulative Time Distribution')
    axes[1, 0].set_xlabel('Time (ms)')
    axes[1, 0].set_ylabel('Cumulative Probability')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # 4. Extra Time Scatter Plot
    min_length = min(len(adaptive_times), len(baseline_times))
    extra_times = adaptive_times[:min_length] - baseline_times[:min_length]
    axes[1, 1].scatter(range(min_length), extra_times, alpha=0.6, s=10)
    axes[1, 1].axhline(
        y=np.mean(extra_times),
        color='r',
        linestyle='--',
        label=f'Avg Extra Time: {np.mean(extra_times):.2f}ms'
    )
    axes[1, 1].set_title('Additional Time per Batch (Adaptive - Baseline)')
    axes[1, 1].set_xlabel('Batch Index')
    axes[1, 1].set_ylabel('Additional Time (ms)')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Reserve space for suptitle

    # Save plot (English filename to avoid encoding issues)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'task_{task_id}_client_{cid}_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.close(fig)  # Release memory to avoid leaks


def print_analysis_results(adaptive_stats: Dict, baseline_stats: Dict, diff_stats: Dict, cid: int):
    """Print analysis results for a single client"""
    print("\n" + "=" * 60)
    print(f"Client {cid} - Time Analysis Results")
    print("=" * 60)

    print("\n--- Baseline Mode Statistics ---")
    for key, value in baseline_stats.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

    print("\n--- Adaptive Mode Statistics ---")
    for key, value in adaptive_stats.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

    print("\n--- Adaptive Mode Overhead ---")
    for key, value in diff_stats.items():
        print(f"{key}: {value:.4f}%" if "%" in key else f"{key}: {value:.4f}")
    print("=" * 60 + "\n")


def find_client_files(data_dir: str, task_id: int) -> Dict[int, Tuple[str, str]]:
    """Auto-find adaptive/baseline file pairs for all clients in a task"""
    # Regex matches: [adaptive/baseline]_task[X]_cid[Y]_batch_times.npy
    pattern = re.compile(r"(adaptive|baseline)_task(\d+)_cid(\d+)_batch_times\.npy", re.IGNORECASE)
    client_files = {}

    for filename in os.listdir(data_dir):
        match = pattern.match(filename)
        if not match:
            continue  # Skip non-matching files

        mode = match.group(1).lower()
        file_task_id = int(match.group(2))
        cid = int(match.group(3))

        # Only process files for the target task
        if file_task_id != task_id:
            continue

        file_path = os.path.join(data_dir, filename)
        # Initialize entry if client not in dict
        if cid not in client_files:
            client_files[cid] = (None, None)  # (adaptive_path, baseline_path)

        # Assign path to corresponding mode
        if mode == "adaptive":
            client_files[cid] = (file_path, client_files[cid][1])
        else:  # baseline
            client_files[cid] = (client_files[cid][0], file_path)

    # Filter out clients with incomplete data (missing adaptive/baseline)
    valid_clients = {
        cid: (adap_path, base_path)
        for cid, (adap_path, base_path) in client_files.items()
        if adap_path is not None and base_path is not None
    }

    print(f"Found {len(valid_clients)}/{len(client_files)} valid clients for Task {task_id}")
    return valid_clients


def main():
    # Configuration (modify these according to your setup)
    TARGET_TASK_ID = 2  # Replace with your task ID
    DATA_DIR = "/home/qiuwenqi/LLM/Fedfinetune/FCL/adaLR/results/adaptive/order_1/outputs/time-compare"  # Replace with your data directory
    SAVE_PLOTS = True  # Set to False if you don't need to save plots

    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)

    # Step 1: Find all valid client data files
    valid_clients = find_client_files(DATA_DIR, TARGET_TASK_ID)
    if not valid_clients:
        print(f"Error: No valid client data found for Task {TARGET_TASK_ID} in {DATA_DIR}")
        return

    # Step 2: Process each client individually
    for cid in sorted(valid_clients.keys()):  # Process clients in order of CID
        print(f"\n----- Processing Client {cid} -----")
        adap_path, base_path = valid_clients[cid]

        # Load data
        adaptive_times, baseline_times = load_time_data(adap_path, base_path)

        # Calculate statistics
        adaptive_stats = calculate_statistics(adaptive_times)
        baseline_stats = calculate_statistics(baseline_times)
        overhead_stats = analyze_time_difference(adaptive_times, baseline_times)

        # Print results
        print_analysis_results(adaptive_stats, baseline_stats, overhead_stats, cid)

        # Generate and save plots (if enabled)
        if SAVE_PLOTS:
            plot_comparison(adaptive_times, baseline_times, TARGET_TASK_ID, cid, save_dir=DATA_DIR)

    print("\nAll clients processed successfully!")


if __name__ == "__main__":
    main()