import numpy as np
from src.solvers import solve_subproblem_qk
import copy


class RPCF:
    """
    Revised Polyhedral Conic Functions (r-PCF) Algorithm.

    This algorithm builds a classification model by iteratively adding polyhedral
    conic functions to separate class -1 (Set A) from class +1 (Set B).
    It uses a "cookie cutter" approach where correctly classified points are removed
    from A at each iteration until A is empty (or maximum iterations reached).
    """

    def __init__(self, C=1.0, lamb=0.01):
        self.C = C
        self.lamb = lamb
        self.functions = []  # List of learned conic functions
        self.centers = []
        self.A_full = None
        self.B_full = None
        self.num_solved_subproblems = 0

    def _evaluate_g(self, X, w, xi, gamma, center):
        """
        Computes the value of the conic function g(x).
        g(x) = w'(x-a) + xi*||x-a||_1 - gamma
        """
        diff = X - center
        term1 = np.dot(diff, w)
        term2 = xi * np.sum(np.abs(diff), axis=1)
        return term1 + term2 - gamma

    def fit(self, X, y):
        if hasattr(self, "estimators_"):
            del self.estimators_  # Clear previous estimators if any

        self.classes_ = np.unique(y)
        if len(self.classes_) > 2:
            self._fit_ova(X, y)
            return

        # --- Binary Classification (Standard RPCF) ---

        # Ensure labels are -1 and 1
        # The user's code in main.py already does this, but for standalone usage:
        # We assume input y is binary. If it's 0/1, map to -1/1?
        # RPCF expects -1 (A) and 1 (B).
        # Let's enforce it locally to be safe, mapping the first class to -1 and second to 1
        # BUT main.py maps 0->-1, 1->1.
        # If we receive 0 and 1, we should map them.
        # If we receive -1 and 1, we are good.

        y_binary = y.copy()
        if set(self.classes_) == {0, 1}:
            y_binary = np.where(y == 0, -1, 1)

        # Separate into A (Class -1) and B (Class 1)
        # Store indices relative to full X
        A_indices = np.where(y_binary == -1)[0].tolist()
        B_indices = np.where(y_binary == 1)[0].tolist()

        self.A_full = X
        self.B_full = X

        # Iteratively separate class A from class B
        iteration = 0
        self.num_solved_subproblems = 0
        while len(A_indices) > 0:
            iteration += 1

            self.current_A_indices = A_indices
            self.current_B_indices = B_indices

            if not A_indices:
                break

            center_idx = self.select_center(A_indices)
            center_a = X[center_idx]

            params = solve_subproblem_qk(
                A_indices, B_indices, X, X, center_a, self.C, self.lamb
            )
            self.num_solved_subproblems += 1

            if params is None:
                print("Solver failed. Break.")
                break

            # Store the model
            model_dict = {**params, "center": center_a}
            self.functions.append(model_dict)
            self.centers.append(center_a)

            # Evaluate to prune datasets
            # For A: keep points where g(a) > 0 (Misclassified/Not covered)
            g_vals_A = self._evaluate_g(
                X[A_indices], params["w"], params["xi"], params["gamma"], center_a
            )
            keep_mask_A = g_vals_A > 0
            A_indices = np.array(A_indices)[keep_mask_A].tolist()

            # For B: keep points where g(b) > 0 (Correctly classified)
            # Note: In original RPCF, set B does not shrink, but some variations allow it.
            # The code already filters B.
            g_vals_B = self._evaluate_g(
                X[B_indices], params["w"], params["xi"], params["gamma"], center_a
            )
            keep_mask_B = g_vals_B > 0
            B_indices = np.array(B_indices)[keep_mask_B].tolist()

            print(
                f"Iter {iteration}: Remaining A: {len(A_indices)}, B: {len(B_indices)}"
            )

    def _fit_ova(self, X, y):
        """
        Trains One-vs-All classifiers for multi-class problems.
        """
        self.estimators_ = []
        print(
            f"Multi-class detected {self.classes_}. Training {len(self.classes_)} OvA estimators..."
        )

        for i, cls in enumerate(self.classes_):
            print(f"  Training Estimator for Class {cls} vs All...")
            # Create binary target: Current Class = 1 (B), All Others = -1 (A)
            # RPCF convention: Separates A (-1) from B (+1).
            # We want to enclose A (-1) with cones?
            # RPCF removes A. So A is the "outside" or "class to be separated".
            # Usually we want to model the POSITIVE class.
            # If RPCF removes A points that are classified as -1 (inside cone?),
            # Wait, RPCF logic:
            # g(x) <= 0  => Class -1 (Inside Cone union)
            # g(x) > 0   => Class +1 (Outside)
            # Wait, let's check basic logic.
            # evaluate_g returns term1 + term2 - gamma.
            # pruning A: keep if g > 0.
            # So points with g <= 0 are REMOVED from A.
            # Means A is correctly classified if g <= 0 (Inside Cone).
            # So RPCF builds a union of cones that covers A (-1).
            # So Class -1 is the "Positive" (Convex/Union of Cones) class in terms of geometric primitives.
            # Class 1 is the "Negative" (Background).

            # For OvA: We typically want to model "Class X" vs "Rest".
            # If we want to detect Class X, we should make Class X = -1 (A), and Rest = 1 (B).
            # So the cones cover Class X.

            y_binary = np.where(y == cls, -1, 1)

            # Create new instance
            # We use copy to preserve init params
            estimator = copy.deepcopy(self)
            # Reset internal state of the copy
            estimator.functions = []
            estimator.centers = []
            estimator.A_full = None
            estimator.B_full = None
            estimator.num_solved_subproblems = 0
            # Remove validation attributes to be safe
            if hasattr(estimator, "classes_"):
                del estimator.classes_
            if hasattr(estimator, "estimators_"):
                del estimator.estimators_

            estimator.fit(X, y_binary)
            self.estimators_.append(estimator)

    def decision_function(self, X):
        """
        Returns scores.
        For Binary: Returns negative of min(g) (Since -1 is A/Inside).
          High positive score => Confident -1 (A).
          High negative score => Confident 1 (B).

          Actually, let's look at predict:
          g_min <= 0 => -1.

          If we want standard sklearn "decision_function":
             Positive => Class 1
             Negative => Class 0 / -1

          Our g_min:
             <= 0 --> Class -1
             > 0  --> Class 1

          So g_min aligns with Class 1 confidence (higher g_min means more likely Class 1).

        For Multi-class (OvA):
          We trained 'Class X' as -1.
          So for estimator i (Class X):
             Score(Class X) should be HIGH if g_min is LOW.
             Because g_min <= 0 means "Predicted as Class X".

          We can define Score_i = -g_min_i.
          Then argmax(Score) picks the class with most negative g val (deepest inside cone).
        """
        if hasattr(self, "estimators_"):
            n_samples = X.shape[0]
            n_classes = len(self.classes_)
            scores = np.zeros((n_samples, n_classes))

            for i, estimator in enumerate(self.estimators_):
                # Estimator i treats Class i as -1.
                # g_vals = estimator.decision_function(X) # Recursive call?
                # No, the sub-estimators are binary instances (estimators_ is deleted).
                # But they are instances of RPCF class.
                g_vals = estimator._compute_g_min(X)
                # g_vals low => Class -1 (Current Class).
                # So Confidence for Class i = -g_vals.
                scores[:, i] = -g_vals
            return scores
        else:
            # Binary
            return self._compute_g_min(X)

    def _compute_g_min(self, X):
        if not self.functions:
            return np.ones(len(X))  # Default to 1 (B)? or 0?

        g_matrix = np.zeros((len(X), len(self.functions)))
        for k, func in enumerate(self.functions):
            g_matrix[:, k] = self._evaluate_g(
                X, func["w"], func["xi"], func["gamma"], func["center"]
            )
        return np.min(g_matrix, axis=1)

    def select_center(self, candidates):
        # Default r-PCF: Random selection
        return np.random.choice(candidates)

    def predict(self, X):
        if hasattr(self, "estimators_"):
            # Multi-class
            scores = self.decision_function(X)
            # argmax gives index of best score
            indices = np.argmax(scores, axis=1)
            return self.classes_[indices]
        else:
            # Binary
            # g_min <= 0 -> -1, else 1
            # Note: We preserved typical RPCF behavior.
            g_min = self._compute_g_min(X)
            # If we trained with 0/1 or labels != -1/1, we should map back.
            # But the binary path maps input to -1/1.
            # We should probably store the binary classes.
            # But standard RPCF just outputs -1/1.
            # Let's check what y was passed.
            return np.where(g_min <= 0, -1, 1)
