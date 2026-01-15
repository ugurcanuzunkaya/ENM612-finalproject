import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, ttest_rel


def calculate_all_significance(results_dict, ds_name="Dataset"):
    """
    Calculates statistical significance between all model pairs.
    Saves results to text file and creates comparison visualizations.

    Args:
        results_dict (dict): Dictionary with model names as keys and accuracy lists as values.
                            e.g., {"rpcf": [...], "vns": [...], "pcf": [...], "svm": [...]}
        ds_name (str): Name of the dataset for logging.

    Returns:
        dict: Dictionary with all pairwise comparison results.
    """
    output_dir = "solutions/stats"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n--- Statistical Significance Tests ({ds_name}) ---")

    all_results = {}
    comparisons = []

    # Define model pairs to compare
    model_pairs = [
        ("rpcf", "vns", "RPCF vs VNS-RPCF"),
        ("rpcf", "pcf", "RPCF vs PCF"),
        # ("rpcf", "svm", "RPCF vs SVM"),
        ("vns", "pcf", "VNS-RPCF vs PCF"),
        # ("vns", "svm", "VNS-RPCF vs SVM"),
        # ("pcf", "svm", "PCF vs SVM"),
    ]

    # Perform pairwise comparisons
    for model1, model2, label in model_pairs:
        res1 = np.array(results_dict.get(model1, []))
        res2 = np.array(results_dict.get(model2, []))

        if len(res1) == 0 or len(res2) == 0 or len(res1) != len(res2):
            continue

        comparison = _compare_pair(res1, res2, label)
        comparisons.append(comparison)

        # Store primary comparison (RPCF vs VNS) as main result
        if model1 == "rpcf" and model2 == "vns":
            all_results.update(comparison)

    all_results["all_comparisons"] = comparisons

    # Save detailed text results
    _save_all_stats(ds_name, results_dict, comparisons, output_dir)

    # Create visualizations
    _plot_all_models_bar(ds_name, results_dict, output_dir)
    _plot_all_models_box(ds_name, results_dict, output_dir)

    return all_results


def _compare_pair(res1, res2, label):
    """Performs statistical tests between two result sets."""
    mean1 = np.mean(res1), np.std(res1)
    mean2 = np.mean(res2), np.std(res2)

    # Wilcoxon test
    try:
        if np.allclose(res1, res2):
            stat_w, p_wilcoxon = 0, 1.0
        else:
            stat_w, p_wilcoxon = wilcoxon(res1, res2)
    except Exception:
        stat_w, p_wilcoxon = 0, 1.0

    # Paired t-test
    try:
        stat_t, p_ttest = ttest_rel(res1, res2)
    except Exception:
        stat_t, p_ttest = 0, 1.0

    # Determine significance
    if p_wilcoxon < 0.05:
        significance = "SIGNIFICANT"
        winner = label.split(" vs ")[1] if mean2 > mean1 else label.split(" vs ")[0]
    else:
        significance = "NOT SIGNIFICANT"
        winner = "No significant difference"

    print(f"  {label}: p={p_wilcoxon:.4f} ({significance})")

    return {
        "label": label,
        "wilcoxon_stat": stat_w,
        "wilcoxon_p": p_wilcoxon,
        "ttest_stat": stat_t,
        "ttest_p": p_ttest,
        "significance": significance,
        "winner": winner,
        "mean1": mean1,
        "mean2": mean2,
    }


def _save_all_stats(ds_name, results_dict, comparisons, output_dir):
    """Saves comprehensive statistical results to text file."""
    txt_filename = f"{output_dir}/{ds_name}_stats.txt"

    with open(txt_filename, "w") as f:
        f.write(f"{'=' * 60}\n")
        f.write(f"    STATISTICAL SIGNIFICANCE TEST RESULTS: {ds_name.upper()}\n")
        f.write(f"{'=' * 60}\n\n")

        # Descriptive statistics for all models
        f.write("--- DESCRIPTIVE STATISTICS ---\n\n")
        f.write(
            f"{'Model':<15} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'N':<5}\n"
        )
        f.write("-" * 60 + "\n")

        model_names = {
            "rpcf": "RPCF",
            "vns": "VNS-RPCF",
            "pcf": "PCF",
            # "svm": "SVM (RBF)",
        }
        for key, name in model_names.items():
            accs = results_dict.get(key, [])
            if accs:
                arr = np.array(accs)
                f.write(
                    f"{name:<15} {np.mean(arr):.4f}     {np.std(arr):.4f}     {np.min(arr):.4f}     {np.max(arr):.4f}     {len(arr)}\n"
                )

        f.write("\n")

        # Pairwise comparisons
        f.write("--- PAIRWISE STATISTICAL COMPARISONS ---\n\n")

        for comp in comparisons:
            f.write(f">> {comp['label']}\n")
            f.write(
                f"   Wilcoxon: stat={comp['wilcoxon_stat']}, p={comp['wilcoxon_p']:.6f}\n"
            )
            f.write(
                f"   t-test:   stat={comp['ttest_stat']:.4f}, p={comp['ttest_p']:.6f}\n"
            )
            f.write(f"   Result:   {comp['significance']}")
            if comp["significance"] == "SIGNIFICANT":
                f.write(f" (Winner: {comp['winner']})")
            f.write("\n\n")

        # Summary
        f.write("--- SUMMARY ---\n\n")
        significant_count = sum(
            1 for c in comparisons if c["significance"] == "SIGNIFICANT"
        )
        f.write(f"Total comparisons: {len(comparisons)}\n")
        f.write(f"Significant differences: {significant_count}\n")

    print(f"  Stats results saved to {txt_filename}")


def _plot_all_models_bar(ds_name, results_dict, output_dir):
    """Creates bar chart comparing all models."""
    plt.figure(figsize=(10, 6))

    model_names = {
        "rpcf": "RPCF",
        "vns": "VNS-RPCF",
        "pcf": "PCF",
    }  # , "svm": "SVM (RBF)"}
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6"]

    names, means, stds = [], [], []
    for key, name in model_names.items():
        accs = results_dict.get(key, [])
        if accs:
            names.append(name)
            means.append(np.mean(accs))
            stds.append(np.std(accs))

    bars = plt.bar(
        names,
        means,
        yerr=stds,
        capsize=8,
        color=colors[: len(names)],
        edgecolor="black",
        linewidth=1.5,
    )

    for bar, mean, std in zip(bars, means, stds):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.01,
            f"{mean:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.ylabel("Accuracy", fontsize=12)
    plt.title(f"Model Comparison: {ds_name}", fontsize=14)
    plt.ylim(0, 1.15)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(f"{output_dir}/{ds_name}_comparison_bar.png", dpi=150)
    plt.close()
    print(f"  Comparison bar chart saved to {output_dir}/{ds_name}_comparison_bar.png")


def _plot_all_models_box(ds_name, results_dict, output_dir):
    """Creates box plot showing distribution for all models."""
    plt.figure(figsize=(10, 6))

    model_names = {
        "rpcf": "RPCF",
        "vns": "VNS-RPCF",
        "pcf": "PCF",
    }  # , "svm": "SVM (RBF)"}
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6"]

    data, labels = [], []
    for key, name in model_names.items():
        accs = results_dict.get(key, [])
        if accs:
            data.append(accs)
            labels.append(name)

    bp = plt.boxplot(data, patch_artist=True, widths=0.5)

    for patch, color in zip(bp["boxes"], colors[: len(labels)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    plt.xticks(range(1, len(labels) + 1), labels, fontsize=11)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title(f"Accuracy Distribution: {ds_name}", fontsize=14)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(f"{output_dir}/{ds_name}_accuracy_distribution.png", dpi=150)
    plt.close()
    print(
        f"  Accuracy distribution plot saved to {output_dir}/{ds_name}_accuracy_distribution.png"
    )


# Keep legacy function for backward compatibility
def calculate_significance(results_rpcf, results_vns, ds_name="Dataset"):
    """
    Legacy function - calculates significance between RPCF and VNS-RPCF only.
    Use calculate_all_significance for comprehensive analysis.
    """
    results_dict = {"rpcf": results_rpcf, "vns": results_vns}
    return calculate_all_significance(results_dict, ds_name)
