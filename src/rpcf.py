import numpy as np
from src.solvers import solve_subproblem_qk


class RPCF:
    def __init__(self, C=1.0, lamb=0.01):
        self.C = C
        self.lamb = lamb
        self.functions = []  # List of dicts
        self.centers = []
        self.A_full = None
        self.B_full = None

    def _evaluate_g(self, X, w, xi, gamma, center):
        # g(x) = w'(x-a) + xi*||x-a||_1 - gamma
        diff = X - center
        term1 = np.dot(diff, w)
        term2 = xi * np.sum(np.abs(diff), axis=1)
        return term1 + term2 - gamma

    def fit(self, X, y):
        # Split into A (Class -1) and B (Class 1)
        # We store indices relative to the FULL X
        A_indices = np.where(y == -1)[0].tolist()
        B_indices = np.where(y == 1)[0].tolist()

        self.A_full = X
        self.B_full = X  # Just a reference to X really, could be cleaner

        # We need to keep track of full X indices for B as well?
        # In r-PCF, B is the set of +1 points. We want to separate A from B.
        # We remove points from A that are correctly classified (g(a) > 0).
        # We assume B is largely kept, but we might remove misclassified B's from constraints
        # to prevent infeasibility or just keep them all?
        # The prompt code says: "In r-PCF, we remove misclassified B points from the constraint set for future iterations"

        iteration = 0
        while len(A_indices) > 0:
            iteration += 1

            # --- EXTENSION POINT: Center Selection ---
            self.current_A_indices = A_indices
            self.current_B_indices = B_indices
            center_idx = self.select_center(A_indices)
            center_a = X[center_idx]
            # -----------------------------------------

            params = solve_subproblem_qk(
                A_indices, B_indices, X, X, center_a, self.C, self.lamb
            )

            if params is None:
                print("Solver failed. Break.")
                break

            # Store Model
            model_dict = {**params, "center": center_a}
            self.functions.append(model_dict)
            self.centers.append(center_a)

            # Evaluate to prune datasets
            # Calculate g(x) for all remaining points in A
            # Correctly classified A points have g(a) > 0 ?
            # Wait, let's re-read the Separation Principle from prompt:
            # "Cover Class -1 (Set A) with negative values... and Class +1 (Set B) with positive values."
            # So A should be g(x) <= 0
            # And B should be g(x) > 0

            # Constraints in solver:
            # Paper Eq 5b: g(x) + 1 <= y_i  => g(x) <= -1 + y_i.  Ideally g(x) <= -1 (negative)
            # Paper Eq 5c: -g(x) + 1 <= z_j => g(x) >= 1 - z_j.   Ideally g(x) >= 1 (positive)

            # So if we successfully classify a point in A, it means g(a) <= 0 (roughly).
            # The prompt code logic says:
            # "Remove points where g(a) <= 0 (Correctly classified as -1)"
            # "Paper: I_{k+1} = {i : g(a) > 0}" -> Keep points where g(a) > 0 (Misclassified or not covered yet)

            # Re-checking prompt code:
            # keep_mask_A = g_vals_A > 0
            # A_indices = np.array(A_indices)[keep_mask_A].tolist()
            # This matches "Keep points where g(a) > 0".
            # Because we want to cover A with negative values, so if g(a) <= 0 it is "covered/removed".

            g_vals_A = self._evaluate_g(
                X[A_indices], params["w"], params["xi"], params["gamma"], center_a
            )
            keep_mask_A = g_vals_A > 0  # Keep if POSITIVE (wrong for A)
            A_indices = np.array(A_indices)[keep_mask_A].tolist()

            # For B:
            # "Keep points where g(b) > 0 (Correctly classified as +1)"
            # "If g(b) <= 0, it is misclassified... remove misclassified B points from constraint set"
            # Prompt code:
            # keep_mask_B = g_vals_B > 0
            # B_indices = np.array(B_indices)[keep_mask_B].tolist()

            g_vals_B = self._evaluate_g(
                X[B_indices], params["w"], params["xi"], params["gamma"], center_a
            )
            keep_mask_B = g_vals_B > 0  # Keep if POSITIVE (correct for B)
            B_indices = np.array(B_indices)[keep_mask_B].tolist()

            print(
                f"Iter {iteration}: Remaining A: {len(A_indices)}, B: {len(B_indices)}"
            )

    def select_center(self, candidates):
        # Default r-PCF: Random selection
        return np.random.choice(candidates)

    def predict(self, X):
        if not self.functions:
            return np.zeros(len(X))

        # g(x) = min(g_1, g_2, ... g_k)
        # Classify as -1 if min(g) <= 0, else 1
        # Wait, if we want to separate A (negative) vs B (positive).
        # If ANY function g_k(x) is negative, then x "falls into" that cone and is classified as A (-1).
        # So taking min is correct logic for union of cones for A.
        # If min(g) <= 0, it belongs to at least one cone -> Class A (-1).
        # If min(g) > 0, it is outside all cones -> Class B (+1).

        g_matrix = np.zeros((len(X), len(self.functions)))

        for k, func in enumerate(self.functions):
            g_matrix[:, k] = self._evaluate_g(
                X, func["w"], func["xi"], func["gamma"], func["center"]
            )

        g_min = np.min(g_matrix, axis=1)
        return np.where(g_min <= 0, -1, 1)
