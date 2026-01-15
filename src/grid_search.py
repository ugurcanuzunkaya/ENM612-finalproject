import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from src.rpcf import RPCF


def grid_search_rpcf(X_train, y_train, X_val, y_val, ds_name=None):
    """
    Performs a simple grid search to find the best hyperparameters (C, lambda)
    for the r-PCF model on a validation set.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        ds_name: Dataset name for saving results (optional)

    Returns:
        dict: A dictionary containing the best 'C' and 'lamb' values found.
    """
    best_acc = -1.0
    best_params = {"C": 1.0, "lamb": 0.01}

    # Range suggested in the paper (simplified for speed)
    C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    lamb_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

    # Store results for visualization
    results = []

    for C in C_values:
        for lamb in lamb_values:
            try:
                model = RPCF(C=C, lamb=lamb)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                acc = accuracy_score(y_val, y_pred)

                results.append({"C": C, "lamb": lamb, "accuracy": acc})

                if acc > best_acc:
                    best_acc = acc
                    best_params = {"C": C, "lamb": lamb}
            except Exception:
                results.append({"C": C, "lamb": lamb, "accuracy": 0.0})
                continue

    print(f"  Best Grid Params: {best_params} (Acc: {best_acc:.4f})")

    # Save grid search results if dataset name is provided
    if ds_name and results:
        save_grid_search_results(
            ds_name, results, best_params, best_acc, C_values, lamb_values
        )

    return best_params


def save_grid_search_results(
    ds_name, results, best_params, best_acc, C_values, lamb_values
):
    """
    Saves grid search results to text file and creates visualization plots.
    """
    output_dir = "solutions/grid_search"
    os.makedirs(output_dir, exist_ok=True)

    # Save text results
    txt_filename = f"{output_dir}/{ds_name}_grid_search.txt"
    with open(txt_filename, "w") as f:
        f.write(f"=== Grid Search Results: {ds_name} ===\n\n")
        f.write(
            f"Best Parameters: C={best_params['C']}, lambda={best_params['lamb']}\n"
        )
        f.write(f"Best Accuracy: {best_acc:.4f}\n\n")
        f.write("All Results:\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'C':<10} {'Lambda':<10} {'Accuracy':<10}\n")
        f.write("-" * 40 + "\n")
        for r in sorted(results, key=lambda x: x["accuracy"], reverse=True):
            f.write(f"{r['C']:<10} {r['lamb']:<10} {r['accuracy']:.4f}\n")

    print(f"  Grid Search results saved to {txt_filename}")

    # Create visualizations
    _plot_grid_search_bar(ds_name, results, best_params, output_dir)
    _plot_accuracy_vs_C(ds_name, results, C_values, lamb_values, output_dir)
    _plot_accuracy_vs_lambda(ds_name, results, C_values, lamb_values, output_dir)


def _plot_grid_search_bar(ds_name, results, best_params, output_dir):
    """
    Creates a bar graph showing accuracy for different hyperparameter combinations.
    """
    # Get top 15 results for readability
    sorted_results = sorted(results, key=lambda x: x["accuracy"], reverse=True)[:15]

    labels = [f"C={r['C']}\nλ={r['lamb']}" for r in sorted_results]
    accuracies = [r["accuracy"] for r in sorted_results]

    # Create color list (highlight best)
    colors = []
    for r in sorted_results:
        if r["C"] == best_params["C"] and r["lamb"] == best_params["lamb"]:
            colors.append("#2ecc71")  # Green for best
        else:
            colors.append("#3498db")  # Blue for others

    plt.figure(figsize=(14, 6))
    bars = plt.bar(range(len(labels)), accuracies, color=colors, edgecolor="black")

    plt.xlabel("Hyperparameter Combination (C, λ)", fontsize=12)
    plt.ylabel("Validation Accuracy", fontsize=12)
    plt.title(
        f"Grid Search Results: {ds_name}\nBest: C={best_params['C']}, λ={best_params['lamb']} (Acc={max(accuracies):.4f})",
        fontsize=13,
    )
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=9)
    plt.ylim(0, 1.0)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        if acc > 0:
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{acc:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    plot_filename = f"{output_dir}/{ds_name}_grid_search_bar.png"
    plt.savefig(plot_filename, dpi=150)
    plt.close()
    print(f"  Grid Search bar graph saved to {plot_filename}")


def _plot_accuracy_vs_C(ds_name, results, C_values, lamb_values, output_dir):
    """
    Creates a line plot showing Accuracy vs C for different fixed lambda values.
    """
    plt.figure(figsize=(10, 6))

    # Use a subset of lambda values for clarity
    lamb_subset = [0.001, 0.01, 0.1, 1, 10]
    colors = plt.cm.viridis([i / len(lamb_subset) for i in range(len(lamb_subset))])

    for idx, lamb in enumerate(lamb_subset):
        if lamb not in lamb_values:
            continue
        accs = []
        for C in C_values:
            match = next(
                (r for r in results if r["C"] == C and r["lamb"] == lamb), None
            )
            accs.append(match["accuracy"] if match else 0.0)

        plt.plot(
            range(len(C_values)),
            accs,
            marker="o",
            label=f"λ={lamb}",
            color=colors[idx],
            linewidth=2,
        )

    plt.xlabel("C (Misclassification Penalty)", fontsize=12)
    plt.ylabel("Validation Accuracy", fontsize=12)
    plt.title(f"Accuracy vs C (Fixed λ) - {ds_name}", fontsize=13)
    plt.xticks(range(len(C_values)), [str(c) for c in C_values], fontsize=10)
    plt.ylim(0, 1.05)
    plt.legend(title="Lambda (λ)", loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_filename = f"{output_dir}/{ds_name}_accuracy_vs_C.png"
    plt.savefig(plot_filename, dpi=150)
    plt.close()
    print(f"  Accuracy vs C plot saved to {plot_filename}")


def _plot_accuracy_vs_lambda(ds_name, results, C_values, lamb_values, output_dir):
    """
    Creates a line plot showing Accuracy vs Lambda for different fixed C values.
    """
    plt.figure(figsize=(10, 6))

    # Use a subset of C values for clarity
    C_subset = [0.01, 0.1, 1, 10, 100]
    colors = plt.cm.plasma([i / len(C_subset) for i in range(len(C_subset))])

    for idx, C in enumerate(C_subset):
        if C not in C_values:
            continue
        accs = []
        for lamb in lamb_values:
            match = next(
                (r for r in results if r["C"] == C and r["lamb"] == lamb), None
            )
            accs.append(match["accuracy"] if match else 0.0)

        plt.plot(
            range(len(lamb_values)),
            accs,
            marker="s",
            label=f"C={C}",
            color=colors[idx],
            linewidth=2,
        )

    plt.xlabel("Lambda (λ) - Regularization", fontsize=12)
    plt.ylabel("Validation Accuracy", fontsize=12)
    plt.title(f"Accuracy vs λ (Fixed C) - {ds_name}", fontsize=13)
    plt.xticks(range(len(lamb_values)), [str(lv) for lv in lamb_values], fontsize=10)
    plt.ylim(0, 1.05)
    plt.legend(title="C", loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_filename = f"{output_dir}/{ds_name}_accuracy_vs_lambda.png"
    plt.savefig(plot_filename, dpi=150)
    plt.close()
    print(f"  Accuracy vs Lambda plot saved to {plot_filename}")


def search_best_k_neighbors(X_train, y_train, X_val, y_val, C, lamb, ds_name=None):
    """
    Searches for the best k_neighbors value for VNS-RPCF.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        C, lamb: Already tuned hyperparameters
        ds_name: Dataset name for saving results (optional)

    Returns:
        int: Best k_neighbors value found.
    """
    from src.vns_rpcf import VNS_RPCF

    # k values to test (subset for speed)
    k_values = [5, 10, 20, 30, 50, 75, 100]

    best_acc = -1.0
    best_k = 20  # Default

    results = []

    print("  Searching for best k_neighbors...", end="", flush=True)

    for k in k_values:
        try:
            model = VNS_RPCF(
                C=C, lamb=lamb, k_neighbors=k, max_vns_iter=5, max_neighbors_check=5
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            acc = accuracy_score(y_val, y_pred)

            results.append({"k": k, "accuracy": acc})

            if acc > best_acc:
                best_acc = acc
                best_k = k
        except Exception:
            results.append({"k": k, "accuracy": 0.0})
            continue

    print(f" Best k={best_k} (Acc: {best_acc:.4f})")

    # Save results if dataset name provided
    if ds_name and results:
        _save_k_search_results(ds_name, results, best_k, best_acc, k_values)

    return best_k


def _save_k_search_results(ds_name, results, best_k, best_acc, k_values):
    """Saves k_neighbors search results to grid_search folder."""
    output_dir = "solutions/grid_search"
    os.makedirs(output_dir, exist_ok=True)

    # Save text results
    txt_filename = f"{output_dir}/{ds_name}_k_search.txt"
    with open(txt_filename, "w") as f:
        f.write(f"=== k_neighbors Search Results: {ds_name} ===\n\n")
        f.write(f"Best k_neighbors: {best_k}\n")
        f.write(f"Best Accuracy: {best_acc:.4f}\n\n")
        f.write("All Results:\n")
        f.write("-" * 30 + "\n")
        f.write(f"{'k':<10} {'Accuracy':<10}\n")
        f.write("-" * 30 + "\n")
        for r in sorted(results, key=lambda x: x["accuracy"], reverse=True):
            f.write(f"{r['k']:<10} {r['accuracy']:.4f}\n")

    # Create bar plot
    plt.figure(figsize=(8, 5))
    k_labels = [str(r["k"]) for r in results]
    accuracies = [r["accuracy"] for r in results]

    colors = ["#2ecc71" if r["k"] == best_k else "#3498db" for r in results]
    bars = plt.bar(k_labels, accuracies, color=colors, edgecolor="black")

    for bar, acc in zip(bars, accuracies):
        if acc > 0:
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{acc:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    plt.xlabel("k_neighbors", fontsize=12)
    plt.ylabel("Validation Accuracy", fontsize=12)
    plt.title(f"k_neighbors Search: {ds_name}\nBest k={best_k}", fontsize=13)
    plt.ylim(0, 1.1)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plot_filename = f"{output_dir}/{ds_name}_k_search_bar.png"
    plt.savefig(plot_filename, dpi=150)
    plt.close()
    print(f"  k_neighbors search results saved to {txt_filename}")


def save_best_params_json(ds_name, params):
    """
    Saves the best parameters to a JSON file.
    Args:
        ds_name (str): Dataset name
        params (dict): Dictionary containing C, lamb, k_opt
    """
    output_dir = "solutions/grid_search"
    os.makedirs(output_dir, exist_ok=True)
    json_filename = f"{output_dir}/{ds_name}_best_params.json"

    with open(json_filename, "w") as f:
        json.dump(params, f, indent=4)

    print(f"  [JSON] Best parameters saved to {json_filename}")
