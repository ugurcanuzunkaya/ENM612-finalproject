import os
import matplotlib.pyplot as plt
import numpy as np
from src.visualizer import plot_decision_boundary


def plot_and_save(model, X, y, title, filename):
    """
    Plots the decision boundary for 2D datasets and saves the figure.
    """
    try:
        if X.shape[1] == 2:
            plot_decision_boundary(model, X, y, title=title)
            plt.savefig(filename)
            plt.close()
            print(f"Plot saved to {filename}")
    except Exception as e:
        print(f"Failed to plot {filename}: {e}")


def save_dataset_results(
    ds_name, overall_stats, last_run_metrics, best_params=None, stats_results=None
):
    """
    Saves the dataset results to two text files:
    1. {ds_name}_results.txt: Summary of hyperparameters, stats, and significance tests.
    2. {ds_name}_detailed_results.txt: All of the above PLUS detailed model parameters.
    """
    results_dir = "solutions/benchmarks"
    os.makedirs(results_dir, exist_ok=True)

    # 1. Summary Results
    summary_filename = f"{results_dir}/{ds_name}_results.txt"
    _write_report(
        summary_filename,
        ds_name,
        overall_stats,
        last_run_metrics,
        best_params,
        stats_results,
        include_details=False,
    )

    # 2. Detailed Results
    detailed_filename = f"{results_dir}/{ds_name}_detailed_results.txt"
    _write_report(
        detailed_filename,
        ds_name,
        overall_stats,
        last_run_metrics,
        best_params,
        stats_results,
        include_details=True,
    )

    print(f"Results saved to {summary_filename}")
    print(f"Detailed Results saved to {detailed_filename}")


def _write_report(
    filename,
    ds_name,
    overall_stats,
    last_run_metrics,
    best_params,
    stats_results,
    include_details=False,
):
    """Helper function to write the report content."""
    with open(filename, "w") as f:
        f.write(f"{'=' * 60}\n")
        f.write(
            f"         {'DETAILED ' if include_details else ''}REPORT FOR DATASET: {ds_name.upper()}\n"
        )
        f.write(f"{'=' * 60}\n\n")

        # --- Best Hyperparameters Section ---
        if best_params:
            f.write("=" * 50 + "\n")
            f.write("           BEST HYPERPARAMETERS (Grid Search)\n")
            f.write("=" * 50 + "\n\n")
            f.write("  RPCF / VNS-RPCF:\n")
            f.write(
                f"    C (Misclassification Penalty): {best_params.get('C', 'N/A')}\n"
            )
            f.write(
                f"    Lambda (Regularization):       {best_params.get('lamb', 'N/A')}\n"
            )
            f.write(
                f"    k_neighbors (VNS):             {best_params.get('k_opt', 'N/A')}\n\n"
            )

        if overall_stats:
            f.write("=" * 50 + "\n")
            f.write("           OVERALL STATISTICS (All Seeds & Folds)\n")
            f.write("=" * 50 + "\n\n")

            header = f"{'Model':<10} {'Mean Test':<10} {'Std Test':<9} {'Mean Train':<11} {'Mean Time':<10} {'Mean Centers':<12}\n"
            f.write(header)
            f.write("-" * len(header) + "\n")

            models_to_report = [("rpcf", "RPCF"), ("vns", "VNS-RPCF"), ("pcf", "PCF")]
            for m_key, m_name in models_to_report:
                stats = overall_stats.get(m_key, {})
                if (
                    isinstance(stats, dict)
                    and "test_acc" in stats
                    and stats["test_acc"]
                ):
                    test_mean = np.mean(stats["test_acc"])
                    test_std = np.std(stats["test_acc"])
                    train_mean = np.mean(stats["train_acc"])
                    time_mean = np.mean(stats["time"])
                    centers_mean = np.mean(stats["centers"])

                    f.write(
                        f"{m_name:<10} {test_mean:<10.4f} {test_std:<9.4f} {train_mean:<11.4f} {time_mean:<10.4f} {centers_mean:<12.1f}\n"
                    )
                else:
                    f.write(
                        f"{m_name:<10} N/A        N/A       N/A         N/A        N/A\n"
                    )

            f.write("\n")

        # --- Statistical Significance Section ---
        if stats_results:
            f.write("=" * 50 + "\n")
            f.write("           STATISTICAL SIGNIFICANCE TESTS\n")
            f.write("=" * 50 + "\n\n")

            all_comparisons = stats_results.get("all_comparisons", [])
            if all_comparisons:
                f.write(
                    f"{'Comparison':<25} {'p-value':<12} {'Result':<20} {'Winner':<20}\n"
                )
                f.write("-" * 77 + "\n")

                for comp in all_comparisons:
                    p_val = comp.get("wilcoxon_p", 1.0)
                    sig = "SIGNIFICANT" if p_val < 0.05 else "NOT SIG."
                    winner = comp.get("winner", "N/A")[:18]
                    f.write(
                        f"{comp.get('label', 'N/A'):<25} {p_val:<12.6f} {sig:<20} {winner:<20}\n"
                    )

                f.write("\n")

            f.write("  Primary Comparison: RPCF vs VNS-RPCF\n\n")
            f.write("  Wilcoxon Signed-Rank Test:\n")
            f.write(f"    Statistic: {stats_results.get('wilcoxon_stat', 'N/A')}\n")
            f.write(f"    p-value:   {stats_results.get('wilcoxon_p', 'N/A'):.6f}\n\n")
            f.write("  Paired t-test:\n")
            f.write(f"    Statistic: {stats_results.get('ttest_stat', 'N/A'):.4f}\n")
            f.write(f"    p-value:   {stats_results.get('ttest_p', 'N/A'):.6f}\n\n")
            f.write(
                f"  Significance (alpha=0.05): {stats_results.get('significance', 'N/A')}\n"
            )
            f.write(f"  Winner: {stats_results.get('winner', 'N/A')}\n\n")

        f.write("=" * 50 + "\n")
        f.write("           DETAILED RESULTS (Last Run Example)\n")
        f.write("=" * 50 + "\n\n")

        for m_key, m_name in [
            ("rpcf", "Standard RPCF"),
            ("vns", "VNS-RPCF"),
            ("pcf", "Original PCF"),
        ]:
            metrics = last_run_metrics.get(m_key, {})
            model = metrics.get("model")

            f.write(f"--- {m_name} ---\n")
            if model and hasattr(model, "functions"):
                f.write(f"Training Time:       {metrics.get('train_time', 0):.4f} s\n")
                f.write(f"Test Time:           {metrics.get('test_time', 0):.4f} s\n")
                f.write(f"Training Accuracy:   {metrics.get('train_acc', 0):.4f}\n")
                f.write(f"Test Accuracy:       {metrics.get('test_acc', 0):.4f}\n")
                f.write(f"Functions (Centers): {len(model.functions)}\n")
                f.write(
                    f"Solved Subproblems:  {getattr(model, 'num_solved_subproblems', 'N/A')}\n"
                )

                if include_details:
                    f.write("\nModel Parameters (Functions):\n")
                    for i, func in enumerate(model.functions):
                        f.write(f"  Function {i + 1}:\n")
                        f.write(f"    Center: {func['center']}\n")
                        f.write(f"    Weight (w): {func['w']}\n")
                        f.write(f"    Xi: {func['xi']:.6f}\n")
                        f.write(f"    Gamma: {func['gamma']:.6f}\n")
                        f.write(f"    QP Objective: {func.get('obj', 'N/A')}\n\n")
            else:
                f.write("Model failed to train or invalid.\n\n")

            f.write("-" * 30 + "\n\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("                     END OF REPORT\n")
        f.write("=" * 60 + "\n")


def save_cv_summary(ds_name, results):
    """
    Saves the aggregated cross-validation results for a dataset.
    """
    filename = f"solutions/benchmarks/{ds_name}_cv_summary.txt"

    with open(filename, "w") as f:
        f.write(f"=== Cross-Validation Summary for Dataset: {ds_name} ===\n\n")

        # RPCF Results
        f.write("--- Standard RPCF ---\n")
        if results["rpcf"]["accuracies"]:
            mean_acc = np.mean(results["rpcf"]["accuracies"])
            std_acc = np.std(results["rpcf"]["accuracies"])
            mean_time = np.mean(results["rpcf"]["times"])
            mean_centers = np.mean(results["rpcf"]["centers"])
            f.write(f"Mean Accuracy: {mean_acc:.4f} (+/- {std_acc:.4f})\n")
            f.write(f"Mean Training Time: {mean_time:.4f} s\n")
            f.write(f"Mean Centers: {mean_centers:.1f}\n")
        else:
            f.write("No successful runs.\n")
        f.write("\n")

        # VNS-RPCF Results
        f.write("--- VNS-RPCF ---\n")
        if results["vns"]["accuracies"]:
            mean_acc = np.mean(results["vns"]["accuracies"])
            std_acc = np.std(results["vns"]["accuracies"])
            mean_time = np.mean(results["vns"]["times"])
            mean_centers = np.mean(results["vns"]["centers"])
            f.write(f"Mean Accuracy: {mean_acc:.4f} (+/- {std_acc:.4f})\n")
            f.write(f"Mean Training Time: {mean_time:.4f} s\n")
            f.write(f"Mean Centers: {mean_centers:.1f}\n")
        else:
            f.write("No successful runs.\n")
        f.write("\n")

        # PCF Results
        f.write("--- Original PCF ---\n")
        if results.get("pcf") and results["pcf"]["accuracies"]:
            mean_acc = np.mean(results["pcf"]["accuracies"])
            std_acc = np.std(results["pcf"]["accuracies"])
            mean_time = np.mean(results["pcf"]["times"])
            mean_centers = np.mean(results["pcf"]["centers"])
            f.write(f"Mean Accuracy: {mean_acc:.4f} (+/- {std_acc:.4f})\n")
            f.write(f"Mean Training Time: {mean_time:.4f} s\n")
            f.write(f"Mean Centers: {mean_centers:.1f}\n")
        else:
            f.write("No successful runs or not implemented.\n")
        f.write("\n")

        # SVM Results
        f.write("--- SVM (RBF) ---\n")
        if results.get("svm") and results["svm"]["accuracies"]:
            mean_acc = np.mean(results["svm"]["accuracies"])
            std_acc = np.std(results["svm"]["accuracies"])
            mean_time = np.mean(results["svm"]["times"])
            f.write(f"Mean Accuracy: {mean_acc:.4f} (+/- {std_acc:.4f})\n")
            f.write(f"Mean Training Time: {mean_time:.4f} s\n")
            # SVM doesn't have "centers" in the same way, maybe support vectors?
            # We track "centers" for SVM as 0 or SV count.
            if results["svm"]["centers"]:
                mean_sv = np.mean(results["svm"]["centers"])
                f.write(f"Mean Support Vectors: {mean_sv:.1f}\n")
        else:
            f.write("No successful runs or not implemented.\n")

    print(f"CV Summary saved to {filename}")


def plot_confusion_matrix(y_true, y_pred, title, filename):
    """
    Plots and saves the confusion matrix.
    """
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[-1, 1])

    plt.figure(figsize=(6, 5))
    disp.plot(cmap=plt.cm.Blues, values_format="d")
    plt.title(title)
    plt.savefig(filename)
    plt.close()
    print(f"Confusion Matrix saved to {filename}")


def plot_vns_convergence(histories, title, filename):
    """
    Plots the convergence of VNS (Score vs Iterations).
    Plots the history of the first function search as a representative example.
    """
    if not histories:
        return

    # Select the longest or first non-empty history
    target_history = histories[0]
    for h in histories:
        if len(h) > len(target_history):
            target_history = h

    plt.figure(figsize=(8, 5))
    plt.plot(
        range(1, len(target_history) + 1),
        target_history,
        marker="o",
        linestyle="-",
        color="b",
    )
    plt.xlabel("VNS Iteration")
    plt.ylabel("Score (Covered A Points)")
    plt.title(title)
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    print(f"VNS Convergence Plot saved to {filename}")
