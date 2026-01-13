"""
r-PCF ve VNS-RPCF Kıyaslamaları için ana yürütme betiği.

Bu betik, standart r-PCF algoritmasını VNS ile geliştirilmiş sürümüyle birden fazla
UCI veri seti üzerinde karşılaştıran kapsamlı bir kıyaslama çalıştırır. Veri yükleme,
ön işleme, hiperparametre ayarlaması için ızgara araması (grid search), eğitim ve
sonuç raporlama işlemlerini yönetir.
"""

import time
import os
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from src.dataloader import DatasetLoader
from src.rpcf import RPCF
from src.vns_rpcf import VNS_RPCF
from src.grid_search import grid_search_rpcf
from src.utils import plot_and_save, save_cv_summary

np.random.seed(42)


def run_all_benchmarks(selected_datasets=None):
    default_datasets = [
        "moons",
        "breast_cancer",
        "blobs_3d",
        "wbcd",
        "wbcp",
        "heart",
        "liver",
        "votes",
        "ionosphere",
    ]

    datasets = selected_datasets if selected_datasets else default_datasets

    loader = DatasetLoader()
    if not os.path.exists("solutions"):
        os.makedirs("solutions")

    print(f"Starting Benchmark Suite on {len(datasets)} datasets with 5-Fold CV...")
    print("=" * 60)

    for ds_name in datasets:
        # ... (rest of the loop remains the same, but since replace_file_content replaces the block, I need to keep the content or just change the definition and the start of the function if I can target it precisely)
        # Actually, replacing the whole function signature and list definition is enough if I use the right TargetContent.
        pass  # Placeholder for thought process

    # Re-reading: I should target the function definition and the list.

    if not os.path.exists("solutions"):
        os.makedirs("solutions")

    print(f"Starting Benchmark Suite on {len(datasets)} datasets with 5-Fold CV...")
    print("=" * 60)

    for ds_name in datasets:
        print(f"\nProcessing Dataset: {ds_name}")
        try:
            X, y = loader.load_dataset(ds_name)
        except Exception as e:
            print(f"Error loading {ds_name}: {e}")
            continue

        # Ensure binary labels are {-1, +1}
        uniques = np.unique(y)
        if set(uniques) == {0, 1}:
            y = np.where(y == 0, -1, 1)
        elif -1 not in uniques:
            min_val = np.min(uniques)
            y = np.where(y == min_val, -1, 1)

        # 5-Fold Stratified Cross-Validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Metrics Storage
        cv_results = {
            "rpcf": {"accuracies": [], "times": [], "centers": []},
            "vns": {"accuracies": [], "times": [], "centers": []},
        }

        fold_idx = 1
        for train_index, test_index in skf.split(X, y):
            print(f"  > Fold {fold_idx}/5...")
            X_train_fold, X_test_fold = X[train_index], X[test_index]
            y_train_fold, y_test_fold = y[train_index], y[test_index]

            # --- Grid Search on Train Fold ---
            # Use a subset of the training fold for validation in grid search
            try:
                X_t, X_v, y_t, y_v = train_test_split(
                    X_train_fold, y_train_fold, test_size=0.2, random_state=42
                )
                best_params = grid_search_rpcf(X_t, y_t, X_v, y_v)
                C_opt = best_params["C"]
                lamb_opt = best_params["lamb"]
            except Exception:
                C_opt, lamb_opt = 10.0, 0.01

            # --- Standard RPCF ---
            start = time.time()
            rpcf = RPCF(C=C_opt, lamb=lamb_opt)
            try:
                rpcf.fit(X_train_fold, y_train_fold)
                t_rpcf = time.time() - start

                y_pred = rpcf.predict(X_test_fold)
                acc = np.mean(y_pred == y_test_fold)
                n_centers = len(rpcf.functions)

                cv_results["rpcf"]["accuracies"].append(acc)
                cv_results["rpcf"]["times"].append(t_rpcf)
                cv_results["rpcf"]["centers"].append(n_centers)
            except Exception as e:
                print(f"    RPCF Fold {fold_idx} Failed: {e}")
                rpcf = None
                t_rpcf = 0

            # --- VNS RPCF ---
            start = time.time()
            vns_rpcf = VNS_RPCF(
                C=C_opt,
                lamb=lamb_opt,
                k_neighbors=20,
                max_vns_iter=5,
                max_neighbors_check=5,
            )
            try:
                vns_rpcf.fit(X_train_fold, y_train_fold)
                t_vns = time.time() - start

                y_pred = vns_rpcf.predict(X_test_fold)
                acc = np.mean(y_pred == y_test_fold)
                n_centers = len(vns_rpcf.functions)

                cv_results["vns"]["accuracies"].append(acc)
                cv_results["vns"]["times"].append(t_vns)
                cv_results["vns"]["centers"].append(n_centers)
            except Exception as e:
                print(f"    VNS Fold {fold_idx} Failed: {e}")
                vns_rpcf = None
                t_vns = 0

            # Optional: Plot last fold for 2D
            if fold_idx == 5 and X.shape[1] == 2:
                if rpcf:
                    plot_and_save(
                        rpcf,
                        X,
                        y,
                        f"RPCF - {ds_name} (Fold 5)",
                        f"solutions/{ds_name}_rpcf_fold5.png",
                    )
                if vns_rpcf:
                    plot_and_save(
                        vns_rpcf,
                        X,
                        y,
                        f"VNS-RPCF - {ds_name} (Fold 5)",
                        f"solutions/{ds_name}_vns_rpcf_fold5.png",
                    )

            fold_idx += 1

        # --- Save Summary Results ---
        save_cv_summary(ds_name, cv_results)

    print("\n" + "=" * 60)
    print("All Benchmarks Completed. Check 'solutions/' for summaries.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Allow passing multiple datasets separated by spaces
        target_datasets = sys.argv[1:]
        run_all_benchmarks(target_datasets)
    else:
        run_all_benchmarks()
