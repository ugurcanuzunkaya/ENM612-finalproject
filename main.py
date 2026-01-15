"""
Main execution script for r-PCF and VNS-RPCF Benchmarking Suite.

This script acts as an all-in-one runner that sequentially executes:
1. Grid Search (Optimization): Finds best C, lambda, k_neighbors.
2. Benchmarks (Execution): Runs models using found parameters and generates reports.

Usage:
    python main.py                  # Run on all default datasets
    python main.py moons heart      # Run on specific datasets
"""

import sys
from run_grid_search import run_grid_search
from run_benchmark import run_benchmarks
from src.sensitivity import run_sensitivity_analysis


def main():
    print("=" * 60)
    print("      ENM612 Final Project - Automated Benchmark Suite")
    print("=" * 60)

    selected_datasets = None
    target_datasets_list = [
        "moons",
        "breast_cancer",
        "blobs_3d",
        "wbcd",
        "wbcp",
        "heart",
        "liver",
        "votes",
        "ionosphere",
        "statlog_heart",
        "abalone",
        "spambase",
    ]

    if len(sys.argv) > 1:
        # Allow passing multiple datasets separated by spaces
        selected_datasets = sys.argv[1:]
        print(f"Target Datasets: {selected_datasets}")
        target_datasets_list = selected_datasets
    else:
        print("Target Datasets: All Default Datasets")

    # --- Phase 1: Grid Search ---
    print("\n" + "=" * 40)
    print(" [Phase 1/3] OPTIMIZATION (Grid Search)")
    print("=" * 40)
    try:
        run_grid_search(selected_datasets)
    except Exception as e:
        print(f"CRITICAL ERROR in Grid Search Phase: {e}")
        return

    # --- Phase 2: Benchmarks ---
    print("\n" + "=" * 40)
    print(" [Phase 2/3] EXECUTION (Benchmarks)")
    print("=" * 40)
    try:
        run_benchmarks(selected_datasets)
    except Exception as e:
        print(f"CRITICAL ERROR in Benchmark Phase: {e}")
        return

    # --- Phase 3: Sensitivity Analysis ---
    print("\n" + "=" * 40)
    print(" [Phase 3/3] ANALYSIS (Sensitivity)")
    print("=" * 40)
    try:
        for ds in target_datasets_list:
            print(f"\nProcessing Dataset: {ds}")
            run_sensitivity_analysis(ds_name=ds)
    except Exception as e:
        print(f"CRITICAL ERROR in Sensitivity Analysis Phase: {e}")
        return

    print("\n" + "=" * 60)
    print("All Phases Completed Successfully.")
    print(
        "Results stored in 'solutions/benchmarks/', 'solutions/grid_search/', and 'solutions/sensitivity/'."
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
