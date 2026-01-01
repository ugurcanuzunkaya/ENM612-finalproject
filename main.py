import time
import os
import numpy as np
from sklearn.model_selection import train_test_split
from src.dataloader import DatasetLoader
from src.rpcf import RPCF
from src.vns_rpcf import VNS_RPCF
from src.grid_search import grid_search_rpcf
from src.utils import plot_and_save, save_dataset_results


def run_all_benchmarks():
    datasets = [
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

    loader = DatasetLoader()
    if not os.path.exists("solutions"):
        os.makedirs("solutions")

    print(f"Starting Benchmark Suite on {len(datasets)} datasets...")
    print("=" * 60)

    for ds_name in datasets:
        print(f"\nProcessing Dataset: {ds_name}")
        try:
            X, y = loader.load_dataset(ds_name)
        except Exception as e:
            print(f"Error loading {ds_name}: {e}")
            continue

        # Preprocessing: Ensure labels are -1 and 1
        uniques = np.unique(y)
        if set(uniques) == {0, 1}:
            y = np.where(y == 0, -1, 1)
        elif -1 not in uniques:
            # If labels are not 0/1, map magnitude-wise or just assume min is -1
            min_val = np.min(uniques)
            y = np.where(y == min_val, -1, 1)

        # Stratified Split
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, stratify=y, random_state=42
            )
        except ValueError:
            # Fallback for small class counts
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )

        # --- Grid Search (Optional but recommended) ---
        print(f"  > Performing Grid Search...")
        # Split train again for val? Or just use CV?
        # For simplicity/speed, we use a fixed validation split from X_train
        try:
            X_t, X_v, y_t, y_v = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
            best_params = grid_search_rpcf(X_t, y_t, X_v, y_v)
            print(f"    Best Params: {best_params}")
            C_opt = best_params["C"]
            lamb_opt = best_params["lamb"]
        except Exception as e:
            print(f"    Grid Search Failed: {e}. Using defaults.")
            C_opt, lamb_opt = 10.0, 0.01

        # --- Standard RPCF ---
        print(f"  > Training Standard RPCF (C={C_opt}, lamb={lamb_opt})...")
        start = time.time()
        rpcf = RPCF(C=C_opt, lamb=lamb_opt)
        try:
            rpcf.fit(X_train, y_train)
            t_rpcf = time.time() - start
            print(f"    Done in {t_rpcf:.2f}s. Centers: {len(rpcf.functions)}")
        except Exception as e:
            print(f"    Failed: {e}")
            rpcf = None
            t_rpcf = 0

        # Plot if 2D
        if rpcf and X.shape[1] == 2:
            plot_and_save(
                rpcf, X, y, f"RPCF - {ds_name}", f"solutions/{ds_name}_rpcf.png"
            )

        # --- VNS RPCF ---
        print(f"  > Training VNS-RPCF (Optimized)..")
        start = time.time()
        # Using same optimal parameters as RPCF for fair comparison?
        # Or should VNS have its own? Usually same.
        vns_rpcf = VNS_RPCF(
            C=C_opt,
            lamb=lamb_opt,
            k_neighbors=20,
            max_vns_iter=5,
            max_neighbors_check=5,
        )
        try:
            vns_rpcf.fit(X_train, y_train)
            t_vns = time.time() - start
            print(f"    Done in {t_vns:.2f}s. Centers: {len(vns_rpcf.functions)}")
        except Exception as e:
            print(f"    Failed: {e}")
            vns_rpcf = None
            t_vns = 0

        # Plot if 2D
        if vns_rpcf and X.shape[1] == 2:
            plot_and_save(
                vns_rpcf,
                X,
                y,
                f"VNS-RPCF - {ds_name}",
                f"solutions/{ds_name}_vns_rpcf.png",
            )

        # --- Save Detailed Results ---
        save_dataset_results(ds_name, X_test, y_test, rpcf, vns_rpcf, t_rpcf, t_vns)

    print("\n" + "=" * 60)
    print("All Benchmarks Completed. Check 'solutions/' directory for results.")


if __name__ == "__main__":
    run_all_benchmarks()
