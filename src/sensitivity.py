import numpy as np
import matplotlib.pyplot as plt
import time
import os
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from src.dataloader import DatasetLoader
from src.vns_rpcf import VNS_RPCF


def run_sensitivity_analysis(ds_name="ionosphere", k_values=None):
    """
    Runs sensitivity analysis on VNS-RPCF k_neighbors parameter.
    Saves text summary and creates combined visualization graphs.
    """
    if k_values is None:
        k_values = [5, 10, 20, 30, 50, 75, 100]

    loader = DatasetLoader()
    print(f"Loading dataset: {ds_name} for sensitivity analysis...")
    try:
        X, y = loader.load_dataset(ds_name)
    except Exception as e:
        print(f"Error loading {ds_name}: {e}")
        return

    # Ensure binary labels are {-1, +1}
    uniques = np.unique(y)
    if set(uniques) == {0, 1}:
        y = np.where(y == 0, -1, 1)
    elif -1 not in uniques:
        min_val = np.min(uniques)
        y = np.where(y == min_val, -1, 1)

    train_accuracies = []
    test_accuracies = []
    train_times = []
    test_times = []
    avg_solved = []
    avg_centers = []

    print(f"Running Sensitivity Analysis on 'k_neighbors': {k_values}")

    # Use a fixed seed for consistency across k values
    np.random.seed(42)
    # Determine Validation Strategy
    n_samples = len(X)
    if n_samples < 1000:
        print(f"  Small dataset (n={n_samples}): Using 10-Fold Stratified CV")
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    else:
        print(f"  Large dataset (n={n_samples}): Using Hold-out (Stratified Split)")
        cv = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    # We will use the average of CV for each k to be robust
    for k in k_values:
        print(f"  Testing k={k}...", end="", flush=True)
        fold_train_accs = []
        fold_test_accs = []
        fold_train_times = []
        fold_test_times = []
        fold_solved = []
        fold_centers = []

        for train_index, test_index in cv.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # VNS parameters (keeping others fixed)
            vns = VNS_RPCF(
                C=10.0, lamb=0.01, k_neighbors=k, max_vns_iter=5, max_neighbors_check=5
            )

            start = time.time()
            try:
                vns.fit(X_train, y_train)
                t_train = time.time() - start

                # Training Accuracy
                y_train_pred = vns.predict(X_train)
                train_acc = accuracy_score(y_train, y_train_pred)

                # Test Time & Accuracy
                t_test_start = time.time()
                y_test_pred = vns.predict(X_test)
                t_test = time.time() - t_test_start
                test_acc = accuracy_score(y_test, y_test_pred)

                # Solved subproblems and centers
                solved = getattr(vns, "num_solved_subproblems", 0)
                centers = len(vns.functions) if hasattr(vns, "functions") else 0

                fold_train_accs.append(train_acc)
                fold_test_accs.append(test_acc)
                fold_train_times.append(t_train)
                fold_test_times.append(t_test)
                fold_solved.append(solved)
                fold_centers.append(centers)

            except Exception as e:
                print(f" (Failed: {e})", end="")

        # Compute Means
        mean_train_acc = np.mean(fold_train_accs) if fold_train_accs else 0.0
        mean_test_acc = np.mean(fold_test_accs) if fold_test_accs else 0.0
        mean_train_time = np.mean(fold_train_times) if fold_train_times else 0.0
        mean_test_time = np.mean(fold_test_times) if fold_test_times else 0.0
        mean_solved = np.mean(fold_solved) if fold_solved else 0.0
        mean_centers = np.mean(fold_centers) if fold_centers else 0.0

        train_accuracies.append(mean_train_acc)
        test_accuracies.append(mean_test_acc)
        train_times.append(mean_train_time)
        test_times.append(mean_test_time)
        avg_solved.append(mean_solved)
        avg_centers.append(mean_centers)

        print(
            f" Avg Test Acc: {mean_test_acc:.4f}, Avg Train Time: {mean_train_time:.4f}s"
        )

    # Save results
    output_dir = "solutions/sensitivity"
    os.makedirs(output_dir, exist_ok=True)

    # Save detailed text summary
    _save_text_summary(
        ds_name,
        k_values,
        train_accuracies,
        test_accuracies,
        train_times,
        test_times,
        avg_solved,
        avg_centers,
        output_dir,
    )

    # Create plots
    _plot_accuracy(ds_name, k_values, test_accuracies, output_dir)
    _plot_time(ds_name, k_values, train_times, output_dir)
    _plot_combined(ds_name, k_values, test_accuracies, train_times, output_dir)
    _plot_metrics_overview(
        ds_name, k_values, test_accuracies, train_times, avg_centers, output_dir
    )


def _save_text_summary(
    ds_name,
    k_values,
    train_accuracies,
    test_accuracies,
    train_times,
    test_times,
    avg_solved,
    avg_centers,
    output_dir,
):
    """
    Saves detailed text summary of sensitivity analysis.
    """
    txt_filename = f"{output_dir}/{ds_name}_sensitivity.txt"
    with open(txt_filename, "w") as f:
        f.write(f"=== Sensitivity Analysis Results: {ds_name} ===\n\n")
        f.write("Parameter: k_neighbors (VNS neighborhood size)\n")
        f.write(f"Tested values: {k_values}\n\n")

        # Find best k
        best_idx = np.argmax(test_accuracies)
        best_k = k_values[best_idx]
        best_acc = test_accuracies[best_idx]
        f.write(f"Best k: {best_k} (Test Accuracy: {best_acc:.4f})\n\n")

        # Summary statistics
        f.write("--- Summary ---\n")
        f.write(
            f"Accuracy Range: [{min(test_accuracies):.4f}, {max(test_accuracies):.4f}]\n"
        )
        f.write(
            f"Time Range:     [{min(train_times):.4f}s, {max(train_times):.4f}s]\n\n"
        )

        # Detailed results table
        f.write("--- Detailed Results ---\n")
        header = f"{'k':<6} {'Train Acc':<12} {'Test Acc':<12} {'Train Time':<12} {'Test Time':<12} {'Centers':<10} {'Solved':<8}\n"
        f.write(header)
        f.write("-" * 72 + "\n")

        for i, k in enumerate(k_values):
            line = (
                f"{k:<6} "
                f"{train_accuracies[i]:<12.4f} "
                f"{test_accuracies[i]:<12.4f} "
                f"{train_times[i]:<12.4f} "
                f"{test_times[i]:<12.4f} "
                f"{avg_centers[i]:<10.2f} "
                f"{avg_solved[i]:<8.2f}\n"
            )
            f.write(line)

    print(f"  Text summary saved to {txt_filename}")


def _plot_accuracy(ds_name, k_values, test_accuracies, output_dir):
    """Creates accuracy vs k_neighbors plot."""
    plt.figure(figsize=(10, 6))
    plt.plot(
        k_values,
        test_accuracies,
        marker="o",
        linestyle="-",
        color="#3498db",
        linewidth=2,
        markersize=8,
    )
    plt.fill_between(k_values, test_accuracies, alpha=0.2, color="#3498db")
    plt.title(f"Sensitivity Analysis: Accuracy vs k_neighbors ({ds_name})", fontsize=14)
    plt.xlabel("k_neighbors", fontsize=12)
    plt.ylabel("Test Accuracy", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{ds_name}_sensitivity_accuracy.png", dpi=150)
    plt.close()
    print(f"  Accuracy plot saved to {output_dir}/{ds_name}_sensitivity_accuracy.png")


def _plot_time(ds_name, k_values, train_times, output_dir):
    """Creates training time vs k_neighbors plot."""
    plt.figure(figsize=(10, 6))
    plt.plot(
        k_values,
        train_times,
        marker="s",
        linestyle="--",
        color="#e74c3c",
        linewidth=2,
        markersize=8,
    )
    plt.fill_between(k_values, train_times, alpha=0.2, color="#e74c3c")
    plt.title(
        f"Sensitivity Analysis: Training Time vs k_neighbors ({ds_name})", fontsize=14
    )
    plt.xlabel("k_neighbors", fontsize=12)
    plt.ylabel("Training Time (s)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{ds_name}_sensitivity_time.png", dpi=150)
    plt.close()
    print(f"  Time plot saved to {output_dir}/{ds_name}_sensitivity_time.png")


def _plot_combined(ds_name, k_values, test_accuracies, train_times, output_dir):
    """Creates combined plot with accuracy and time on dual y-axes."""
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Accuracy on left axis
    color1 = "#3498db"
    ax1.set_xlabel("k_neighbors", fontsize=12)
    ax1.set_ylabel("Test Accuracy", color=color1, fontsize=12)
    line1 = ax1.plot(
        k_values,
        test_accuracies,
        marker="o",
        linestyle="-",
        color=color1,
        linewidth=2,
        markersize=8,
        label="Accuracy",
    )
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)

    # Time on right axis
    ax2 = ax1.twinx()
    color2 = "#e74c3c"
    ax2.set_ylabel("Training Time (s)", color=color2, fontsize=12)
    line2 = ax2.plot(
        k_values,
        train_times,
        marker="s",
        linestyle="--",
        color=color2,
        linewidth=2,
        markersize=8,
        label="Time",
    )
    ax2.tick_params(axis="y", labelcolor=color2)

    # Combined legend
    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper right", fontsize=11)

    plt.title(
        f"Sensitivity Analysis: Accuracy & Time vs k_neighbors ({ds_name})", fontsize=14
    )
    fig.tight_layout()
    plt.savefig(f"{output_dir}/{ds_name}_sensitivity_combined.png", dpi=150)
    plt.close()
    print(f"  Combined plot saved to {output_dir}/{ds_name}_sensitivity_combined.png")


def _plot_metrics_overview(
    ds_name, k_values, test_accuracies, train_times, avg_centers, output_dir
):
    """Creates an overview with subplots for all metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Accuracy
    axes[0].plot(k_values, test_accuracies, marker="o", color="#3498db", linewidth=2)
    axes[0].set_title("Test Accuracy", fontsize=12)
    axes[0].set_xlabel("k_neighbors")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1.05)

    # Training Time
    axes[1].plot(k_values, train_times, marker="s", color="#e74c3c", linewidth=2)
    axes[1].set_title("Training Time", fontsize=12)
    axes[1].set_xlabel("k_neighbors")
    axes[1].set_ylabel("Time (s)")
    axes[1].grid(True, alpha=0.3)

    # Number of Centers
    axes[2].plot(k_values, avg_centers, marker="^", color="#2ecc71", linewidth=2)
    axes[2].set_title("Model Complexity (Centers)", fontsize=12)
    axes[2].set_xlabel("k_neighbors")
    axes[2].set_ylabel("Avg. Centers")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(
        f"VNS-RPCF Sensitivity Analysis Overview: {ds_name}", fontsize=14, y=1.02
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{ds_name}_sensitivity_overview.png", dpi=150)
    plt.close()
    print(f"  Overview plot saved to {output_dir}/{ds_name}_sensitivity_overview.png")


if __name__ == "__main__":
    import argparse

    DEFAULT_DATASETS = [
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

    parser = argparse.ArgumentParser(
        description="Run Sensitivity Analysis on VNS-RPCF k_neighbors."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default="ionosphere",
        help="Name of the dataset to analyze, or 'all' for all datasets.",
    )

    args = parser.parse_args()

    if args.dataset.lower() == "all":
        print(f"Running sensitivity analysis on ALL datasets: {DEFAULT_DATASETS}")
        for ds in DEFAULT_DATASETS:
            print(f"\n--- Processing {ds} ---")
            run_sensitivity_analysis(ds_name=ds)
    else:
        run_sensitivity_analysis(ds_name=args.dataset)
