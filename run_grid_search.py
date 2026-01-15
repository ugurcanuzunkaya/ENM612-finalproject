import sys
import os
import numpy as np
from sklearn.model_selection import train_test_split
from src.dataloader import DatasetLoader
from src.grid_search import (
    grid_search_rpcf,
    search_best_k_neighbors,
    save_best_params_json,
)

SEEDS = [42, 10, 123, 2024, 7]


def run_grid_search(selected_datasets=None):
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
    os.makedirs("solutions/grid_search", exist_ok=True)

    print(f"Starting Grid Search on {len(datasets)} datasets...")
    print("=" * 60)

    for ds_name in datasets:
        print(f"\nProcessing Dataset: {ds_name} for Hyperparameter Tuning")
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

        # For Grid Search, we only need a representative split, usually from the first seed
        seed = SEEDS[
            -1
        ]  # Use the last seed as in the original main.py logic for saving
        np.random.seed(seed)

        # Use a simple train/val split for grid search to keep it fast but robust enough
        # The original code did grid search on the *train fold* of a CV split.
        # To make this robust, let's just use a 80/20 split of the whole dataset.
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )

        print(f"  > Running Grid Search (RPCF)...")
        # Run RPCF Grid Search
        best_params_rpcf = grid_search_rpcf(
            X_train, y_train, X_val, y_val, ds_name=ds_name
        )
        C_opt = best_params_rpcf["C"]
        lamb_opt = best_params_rpcf["lamb"]

        print(f"  > Searching for best k_neighbors (VNS-RPCF)...")
        # Run k_neighbors Search
        k_opt = search_best_k_neighbors(
            X_train, y_train, X_val, y_val, C_opt, lamb_opt, ds_name=ds_name
        )

        # Compile all params
        all_best_params = {"C": C_opt, "lamb": lamb_opt, "k_opt": k_opt}

        # Save to JSON
        save_best_params_json(ds_name, all_best_params)

    print("\n" + "=" * 60)
    print("Grid Search Modular Execution Completed.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_datasets = sys.argv[1:]
        run_grid_search(target_datasets)
    else:
        run_grid_search()
