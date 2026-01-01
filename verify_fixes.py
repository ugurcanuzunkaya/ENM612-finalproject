import numpy as np
import time
from src.solvers import solve_subproblem_qk
from src.vns_rpcf import VNS_RPCF
from src.dataloader import DatasetLoader
from src.rpcf import RPCF


def verify_empty_b():
    print("--- Verifying Empty B Case ---")
    # Simulate a case where A has points but B is empty
    X = np.random.randn(10, 2)
    center = X[0]
    A_indices = list(range(10))
    B_indices = []  # Empty!

    print("Calling solver with empty B...")
    try:
        params = solve_subproblem_qk(
            A_indices, B_indices, X, X, center, C=1.0, lamb=0.01
        )
        if params is not None:
            print("SUCCESS: Solver returned valid parameters even with empty B.")
            print(f"Obj: {params['obj']}")
        else:
            print("FAILURE: Solver returned None for empty B.")
    except Exception as e:
        print(f"FAILURE: Exception raised: {e}")


def verify_liver_dataset():
    print("\n--- Verifying Liver Dataset (Imbalance) ---")
    # Liver dataset was problematic. Let's load it and run a quick check.
    loader = DatasetLoader()
    try:
        X, y = loader.load_dataset("liver")
        if set(np.unique(y)) != {-1, 1}:
            y = np.where(
                y == 1, 1, -1
            )  # specific to liver might need check, but let's assume standard

        print(
            f"Liver Data: {X.shape}, Class -1: {np.sum(y == -1)}, Class 1: {np.sum(y == 1)}"
        )

        # Train generic RPCF
        model = RPCF(C=10.0, lamb=0.01)
        model.fit(X, y)

        # Check predictions
        preds = model.predict(X)
        acc = np.mean(preds == y)
        recall_neg1 = np.sum((preds == -1) & (y == -1)) / np.sum(y == -1)

        print(f"Accuracy: {acc:.4f}")
        print(f"Recall (Class -1): {recall_neg1:.4f}")

        if recall_neg1 > 0.0:
            print("SUCCESS: Class -1 is being predicted (Recall > 0).")
        else:
            print("FAILURE: Class -1 Recall is 0.0 (Model ignoring minority class).")

    except Exception as e:
        print(f"Could not load liver dataset or other error: {e}")


def verify_vns_limit():
    print("\n--- Verifying VNS Neighbor Limit ---")
    # Use simple moons dataset
    loader = DatasetLoader()
    try:
        X, y = loader.load_dataset("moons")

        print("Training VNS with limit=2...")
        start = time.time()
        vns = VNS_RPCF(
            C=10.0, lamb=0.01, k_neighbors=20, max_vns_iter=2, max_neighbors_check=2
        )
        vns.fit(X, y)
        end = time.time()
        print(f"VNS (Limit=2) Time: {end - start:.4f}s")

        print("Training VNS with limit=20 (Simulating old behavior)...")
        start = time.time()
        vns_full = VNS_RPCF(
            C=10.0, lamb=0.01, k_neighbors=20, max_vns_iter=2, max_neighbors_check=20
        )
        vns_full.fit(X, y)
        end = time.time()
        print(f"VNS (Limit=20) Time: {end - start:.4f}s")

    except Exception as e:
        print(f"VNS check failed: {e}")


if __name__ == "__main__":
    verify_empty_b()
    verify_liver_dataset()
    verify_vns_limit()
