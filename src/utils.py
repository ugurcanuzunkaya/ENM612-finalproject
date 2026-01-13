import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from src.visualizer import plot_decision_boundary


def plot_and_save(model, X, y, title, filename):
    """
    2D veri setleri için karar sınırını çizer ve şekli kaydeder.
    """
    try:
        if X.shape[1] == 2:
            plot_decision_boundary(model, X, y, title=title)
            plt.savefig(filename)
            plt.close()
            print(f"Plot saved to {filename}")
    except Exception as e:
        print(f"Failed to plot {filename}: {e}")


def save_dataset_results(ds_name, X_test, y_test, rpcf_model, vns_model, t_rpcf, t_vns):
    """
    İlk olarak veri seti için genel sonuçları ve özeti kaydeder.
    Sonra ayrıntılı model sonuçlarını, parametrelerini ve metriklerini bir metin dosyasına kaydeder.
    """
    filename = f"solutions/{ds_name}_results.txt"

    with open(filename, "w") as f:
        f.write(f"=== Summary for Dataset: {ds_name} ===\n\n")
        if rpcf_model and hasattr(rpcf_model, "functions"):
            f.write("Standard RPCF: ")
            y_pred = rpcf_model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"Training Time: {t_rpcf:.4f} seconds\n")
            f.write(f"Number of Functions (Centers): {len(rpcf_model.functions)}\n")

        if vns_model and hasattr(vns_model, "functions"):
            f.write("VNS-RPCF: ")
            y_pred = vns_model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"Training Time: {t_vns:.4f} seconds\n")
            f.write(f"Number of Functions (Centers): {len(vns_model.functions)}\n")

        f.write("\n")
        f.write(f"=== Detailed Results for Dataset: {ds_name} ===\n\n")

        # --- RPCF Bölümü ---
        f.write("--- Model: Standard RPCF ---\n")
        if rpcf_model and hasattr(rpcf_model, "functions"):
            y_pred = rpcf_model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"Training Time: {t_rpcf:.4f} seconds\n")
            f.write(f"Number of Functions (Centers): {len(rpcf_model.functions)}\n")
            f.write("\nEvaluation Metrics:\n")
            f.write(classification_report(y_test, y_pred, zero_division=0))

            f.write("\nModel Parameters (Functions):\n")
            for i, func in enumerate(rpcf_model.functions):
                f.write(f"  Function {i + 1}:\n")
                f.write(f"    Center: {func['center']}\n")
                f.write(f"    Weight (w): {func['w']}\n")
                f.write(f"    Xi: {func['xi']:.6f}\n")
                f.write(f"    Gamma: {func['gamma']:.6f}\n")
                f.write(f"    QP Objective: {func.get('obj', 'N/A')}\n\n")
        else:
            f.write("Model failed to train or invalid.\n\n")

        f.write("-" * 50 + "\n\n")

        # --- VNS-RPCF Bölümü ---
        f.write("--- Model: VNS-RPCF ---\n")
        if vns_model and hasattr(vns_model, "functions"):
            y_pred = vns_model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"Training Time: {t_vns:.4f} seconds\n")
            f.write(f"Number of Functions (Centers): {len(vns_model.functions)}\n")
            f.write("\nEvaluation Metrics:\n")
            f.write(classification_report(y_test, y_pred, zero_division=0))

            f.write("\nModel Parameters (Functions):\n")
            for i, func in enumerate(vns_model.functions):
                f.write(f"  Function {i + 1}:\n")
                f.write(f"    Center: {func['center']}\n")
                f.write(f"    Weight (w): {func['w']}\n")
                f.write(f"    Xi: {func['xi']:.6f}\n")
                f.write(f"    Gamma: {func['gamma']:.6f}\n")
                f.write(f"    QP Objective: {func.get('obj', 'N/A')}\n\n")
        else:
            f.write("Model failed to train or invalid.\n\n")

    print(f"Results saved to {filename}")


def save_cv_summary(ds_name, results):
    """
    Saves the aggregated cross-validation results for a dataset.
    """
    filename = f"solutions/{ds_name}_cv_summary.txt"

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

    print(f"CV Summary saved to {filename}")
