import time
import os
import json
import numpy as np
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedShuffleSplit,
)
from sklearn.metrics import accuracy_score
from src.dataloader import DatasetLoader
from src.rpcf import RPCF
from src.vns_rpcf import VNS_RPCF
from src.pcf import PCF
from src.utils import (
    plot_and_save,
    save_dataset_results,
    plot_confusion_matrix,
    plot_vns_convergence,
)
from src.visualizer import plot_decision_boundary_3d
# from src.stats import calculate_all_significance # Imported locally in function

SEEDS = [42, 10, 123, 2024, 7]


def run_benchmarks(selected_datasets=None):
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
        "statlog_heart",
        "abalone",
        "spambase",
    ]

    datasets = selected_datasets if selected_datasets else default_datasets

    loader = DatasetLoader()
    # Create output directories
    os.makedirs("solutions/benchmarks", exist_ok=True)
    os.makedirs("solutions/stats", exist_ok=True)

    # Ensure grid search dir exists to check for params
    os.makedirs("solutions/grid_search", exist_ok=True)

    print(f"Starting Benchmark Execution on {len(datasets)} datasets...")
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

        # Load Best Params
        json_path = f"solutions/grid_search/{ds_name}_best_params.json"
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                best_params = json.load(f)
            C_opt = best_params.get("C", 10.0)
            lamb_opt = best_params.get("lamb", 0.01)
            k_opt = best_params.get("k_opt", 20)
            print(f"  > Loaded Parameters: C={C_opt}, lambda={lamb_opt}, k={k_opt}")
        else:
            print(f"  > [Warning] No parameters found at {json_path}. Using defaults.")
            best_params = {"C": 10.0, "lamb": 0.01, "k_opt": 20}
            C_opt, lamb_opt, k_opt = 10.0, 0.01, 20

        ds_seed_stats = {
            "rpcf": {"test_acc": [], "train_acc": [], "time": [], "centers": []},
            "vns": {"test_acc": [], "train_acc": [], "time": [], "centers": []},
            "pcf": {"test_acc": [], "train_acc": [], "time": [], "centers": []},
        }

        last_train_data = None
        last_test_data = None

        # We need to capture the last models for reporting
        last_models = {"rpcf": None, "vns": None, "pcf": None}

        for seed in SEEDS:
            # print(f"  > Seed: {seed}")
            np.random.seed(seed)

            n_samples = len(X)
            if n_samples < 1000:
                # print(f"    - Small dataset (n={n_samples}): Using 10-Fold Stratified CV")
                cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
                n_folds_total = 10
            else:
                # print(f"    - Large dataset (n={n_samples}): Using Hold-out")
                cv = StratifiedShuffleSplit(
                    n_splits=1, test_size=0.2, random_state=seed
                )
                n_folds_total = 1

            seed_results = {
                "rpcf": {"test_acc": [], "train_acc": [], "time": [], "centers": []},
                "vns": {"test_acc": [], "train_acc": [], "time": [], "centers": []},
                "pcf": {"test_acc": [], "train_acc": [], "time": [], "centers": []},
            }

            fold_idx = 1
            for train_index, test_index in cv.split(X, y):
                X_train_fold, X_test_fold = X[train_index], X[test_index]
                y_train_fold, y_test_fold = y[train_index], y[test_index]

                # --- Standard RPCF ---
                start = time.time()
                rpcf = RPCF(C=C_opt, lamb=lamb_opt)
                rpcf_success = False
                try:
                    rpcf.fit(X_train_fold, y_train_fold)
                    t_rpcf = time.time() - start
                    tr_acc = np.mean(rpcf.predict(X_train_fold) == y_train_fold)
                    acc = np.mean(rpcf.predict(X_test_fold) == y_test_fold)
                    n_centers = len(rpcf.functions)

                    seed_results["rpcf"]["test_acc"].append(acc)
                    seed_results["rpcf"]["train_acc"].append(tr_acc)
                    seed_results["rpcf"]["time"].append(t_rpcf)
                    seed_results["rpcf"]["centers"].append(n_centers)
                    rpcf_success = True
                    last_models["rpcf"] = rpcf
                except Exception as e:
                    print(f"      RPCF Fold {fold_idx} Failed: {e}")
                    t_rpcf = 0

                # --- VNS RPCF ---
                start = time.time()
                vns_rpcf = VNS_RPCF(
                    C=C_opt,
                    lamb=lamb_opt,
                    k_neighbors=k_opt,
                    max_vns_iter=5,
                    max_neighbors_check=5,
                )
                vns_success = False
                try:
                    vns_rpcf.fit(X_train_fold, y_train_fold)
                    t_vns = time.time() - start
                    tr_acc_vns = np.mean(vns_rpcf.predict(X_train_fold) == y_train_fold)
                    acc = np.mean(vns_rpcf.predict(X_test_fold) == y_test_fold)
                    n_centers = len(vns_rpcf.functions)

                    seed_results["vns"]["test_acc"].append(acc)
                    seed_results["vns"]["train_acc"].append(tr_acc_vns)
                    seed_results["vns"]["time"].append(t_vns)
                    seed_results["vns"]["centers"].append(n_centers)
                    vns_success = True
                    last_models["vns"] = vns_rpcf
                except Exception as e:
                    print(f"      VNS Fold {fold_idx} Failed: {e}")
                    t_vns = 0

                # --- PCF ---
                start = time.time()
                pcf_model = PCF()
                pcf_success = False
                try:
                    pcf_model.fit(X_train_fold, y_train_fold)
                    t_pcf = time.time() - start
                    tr_acc_pcf = np.mean(
                        pcf_model.predict(X_train_fold) == y_train_fold
                    )
                    acc_pcf = np.mean(pcf_model.predict(X_test_fold) == y_test_fold)
                    n_centers_pcf = len(pcf_model.functions)

                    seed_results["pcf"]["test_acc"].append(acc_pcf)
                    seed_results["pcf"]["train_acc"].append(tr_acc_pcf)
                    seed_results["pcf"]["time"].append(t_pcf)
                    seed_results["pcf"]["centers"].append(n_centers_pcf)
                    pcf_success = True
                    last_models["pcf"] = pcf_model
                except Exception as e:
                    print(f"      PCF Fold {fold_idx} Failed: {e}")
                    t_pcf = 0

                # --- Visualizations (Last Fold & Last Seed Only) ---
                if fold_idx == n_folds_total and seed == SEEDS[-1]:
                    if vns_success:
                        y_pred_vns = vns_rpcf.predict(X_test_fold)
                        plot_confusion_matrix(
                            y_test_fold,
                            y_pred_vns,
                            f"Confusion Matrix: VNS-RPCF ({ds_name})",
                            f"solutions/benchmarks/{ds_name}_cm_vns_fold{fold_idx}.png",
                        )
                        if hasattr(vns_rpcf, "all_vns_histories"):
                            plot_vns_convergence(
                                vns_rpcf.all_vns_histories,
                                f"VNS Convergence ({ds_name})",
                                f"solutions/benchmarks/{ds_name}_vns_conv_fold{fold_idx}.png",
                            )

                    if X.shape[1] == 2:
                        if rpcf_success:
                            plot_and_save(
                                rpcf,
                                X,
                                y,
                                f"RPCF - {ds_name}",
                                f"solutions/benchmarks/{ds_name}_rpcf.png",
                            )
                        if vns_success:
                            plot_and_save(
                                vns_rpcf,
                                X,
                                y,
                                f"VNS-RPCF - {ds_name}",
                                f"solutions/benchmarks/{ds_name}_vns_rpcf.png",
                            )
                    elif X.shape[1] == 3:
                        if rpcf_success:
                            plot_decision_boundary_3d(
                                rpcf,
                                X,
                                y,
                                f"RPCF - {ds_name}",
                                f"solutions/benchmarks/{ds_name}_rpcf_3d.png",
                            )
                        if vns_success:
                            plot_decision_boundary_3d(
                                vns_rpcf,
                                X,
                                y,
                                f"VNS-RPCF - {ds_name}",
                                f"solutions/benchmarks/{ds_name}_vns_rpcf_3d.png",
                            )

                fold_idx += 1
                last_train_data = (X_train_fold, y_train_fold)
                last_test_data = (X_test_fold, y_test_fold)

            for model_key in ["rpcf", "vns", "pcf"]:
                for metric in ["test_acc", "train_acc", "time", "centers"]:
                    ds_seed_stats[model_key][metric].extend(
                        seed_results[model_key][metric]
                    )

        # --- End of Seeds Loop ---
        print(
            f"  > Summary: RPCF Acc: {np.mean(ds_seed_stats['rpcf']['test_acc']):.4f} | VNS Acc: {np.mean(ds_seed_stats['vns']['test_acc']):.4f}"
        )

        # Prepare Metrics for Report
        last_run_metrics = {"rpcf": {}, "vns": {}, "pcf": {}}
        if last_train_data and last_test_data:
            X_tr, y_tr = last_train_data
            X_te, y_te = last_test_data

            for m_key in ["rpcf", "vns", "pcf"]:
                model = last_models[m_key]
                if model:
                    # Re-eval for timing/accuracy on last fold explicitly if needed, but we already have it from loop?
                    # Ideally we just take the last loop's values, but 't_rpcf' variable scope is tricky.
                    # Let's just re-predict to be safe for the report or use the gathered stats.
                    # The report function expects a dict with train_time, test_time, etc.
                    # Since we didn't store per-fold times in `last_models`, we can just grab mean or calc new.
                    # Let's calc fresh metrics on the last model object to be precise for the "Example Run".

                    start = time.time()
                    tr_pred = model.predict(X_tr)
                    tr_acc = accuracy_score(y_tr, tr_pred)  # Re-verify

                    t0_test = time.time()
                    te_pred = model.predict(X_te)
                    test_time = time.time() - t0_test
                    te_acc = accuracy_score(y_te, te_pred)

                    # Training time is lost, we'll estimate from average or just use 0.0 if not stored
                    # Actually we can use the last `t_rpcf` etc if we initialized them outside loop or stored in a list.
                    # We have `ds_seed_stats['rpcf']['time']` list. Let's use the last element.
                    last_train_time = (
                        ds_seed_stats[m_key]["time"][-1]
                        if ds_seed_stats[m_key]["time"]
                        else 0.0
                    )

                    last_run_metrics[m_key] = {
                        "model": model,
                        "train_time": last_train_time,
                        "test_time": test_time,
                        "train_acc": tr_acc,
                        "test_acc": te_acc,
                    }

        # Statistical Significance
        stats_results = {}
        if ds_seed_stats["rpcf"]["test_acc"] and ds_seed_stats["vns"]["test_acc"]:
            from src.stats import calculate_all_significance

            stats_results = calculate_all_significance(
                {
                    "rpcf": ds_seed_stats["rpcf"]["test_acc"],
                    "vns": ds_seed_stats["vns"]["test_acc"],
                    "pcf": ds_seed_stats["pcf"]["test_acc"],
                },
                ds_name,
            )

        # Save Reports
        save_dataset_results(
            ds_name, ds_seed_stats, last_run_metrics, best_params, stats_results
        )

    print("\n" + "=" * 60)
    print("Benchmark Execution Completed.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        target_datasets = sys.argv[1:]
        run_benchmarks(target_datasets)
    else:
        run_benchmarks()
