import numpy as np
from src.solvers import solve_subproblem_qk


class RPCF:
    """
    Revised Polyhedral Conic Functions (r-PCF) Algorithm.

    This algorithm constructs a classification model by iteratively adding
    polyhedral conic functions to separate class -1 (Set A) from class +1 (Set B).
    It uses a "cookie-cutter" approach where correctly classified points from A are
    removed in each iteration until A is empty (or max iterations reached).
    """

    def __init__(self, C=1.0, lamb=0.01):
        self.C = C
        self.lamb = lamb
        self.functions = []  # List of learned conic functions
        self.centers = []
        self.A_full = None
        self.B_full = None

    def _evaluate_g(self, X, w, xi, gamma, center):
        """
        Calculates the value of the conic function g(x).
        g(x) = w'(x-a) + xi*||x-a||_1 - gamma
        """
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
        self.B_full = X

        # Iteratively separate class A from class B.
        # Ideally, we want to find a set of cones whose intersection classifies B correctly
        # and excludes A. In the constructive approach, we remove points from A that are
        # "cut" (correctly classified as outside the current cone) in each step.

        iteration = 0
        while len(A_indices) > 0:
            iteration += 1

            self.current_A_indices = A_indices
            self.current_B_indices = B_indices
            center_idx = self.select_center(A_indices)
            center_a = X[center_idx]

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
            # For A: Keep points where g(a) > 0 (Misclassified/Not covered)
            g_vals_A = self._evaluate_g(
                X[A_indices], params["w"], params["xi"], params["gamma"], center_a
            )
            keep_mask_A = g_vals_A > 0
            A_indices = np.array(A_indices)[keep_mask_A].tolist()

            # For B: Keep points where g(b) > 0 (Correctly classified)
            g_vals_B = self._evaluate_g(
                X[B_indices], params["w"], params["xi"], params["gamma"], center_a
            )
            keep_mask_B = g_vals_B > 0
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
        g_matrix = np.zeros((len(X), len(self.functions)))

        for k, func in enumerate(self.functions):
            g_matrix[:, k] = self._evaluate_g(
                X, func["w"], func["xi"], func["gamma"], func["center"]
            )

        g_min = np.min(g_matrix, axis=1)
        return np.where(g_min <= 0, -1, 1)
