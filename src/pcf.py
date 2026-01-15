import numpy as np
from src.solvers import solve_subproblem_pk
import copy


class PCF:
    """
    Original Polyhedral Conic Functions (PCF) Algorithm.
    Based on Gasimov & Ozturk (2006) / Ceylan & Ozturk (2020).

    Differences from RPCF:
    - Hard constraint for B (Set B cannot be misclassified).
    - No regularization term.
    - Only Class A (-1) is covered by conic functions (Separation).
    """

    def __init__(self):
        # PCF has no C and lambda hyperparameters.
        self.functions = []
        self.centers = []
        self.A_full = None
        self.B_full = None
        self.num_solved_subproblems = 0
        self.classes_ = None

    def _evaluate_g(self, X, w, xi, gamma, center):
        """
        g(x) = w'(x-a) + xi*||x-a||_1 - gamma
        """
        diff = X - center
        term1 = np.dot(diff, w)
        term2 = xi * np.sum(np.abs(diff), axis=1)
        return term1 + term2 - gamma

    def fit(self, X, y):
        # Clear previous state
        if hasattr(self, "estimators_"):
            del self.estimators_
        self.functions = []
        self.centers = []

        self.classes_ = np.unique(y)
        if len(self.classes_) > 2:
            self._fit_ova(X, y)
            return

        # --- Binary Classification ---
        # Ensure labels are -1 and 1
        y_binary = y.copy()
        if set(self.classes_) == {0, 1}:
            y_binary = np.where(y == 0, -1, 1)

        A_indices = np.where(y_binary == -1)[0].tolist()
        B_indices = np.where(y_binary == 1)[0].tolist()

        self.A_full = X
        self.B_full = X

        iteration = 0
        self.num_solved_subproblems = 0

        # PCF Loop
        while len(A_indices) > 0:
            iteration += 1
            if not A_indices:
                break

            center_idx = self.select_center(A_indices)
            center_a = X[center_idx]

            # Solve Subproblem P_k
            params = solve_subproblem_pk(A_indices, B_indices, X, X, center_a)
            self.num_solved_subproblems += 1

            if params is None:
                print("PCF Solver failed. Break.")
                break

            # Store Model
            model_dict = {**params, "center": center_a}
            self.functions.append(model_dict)
            self.centers.append(center_a)

            # Prune A: Remove points that are now correctly classified (g(a) <= 0 ideally?)
            # Wait, in RPCF we kept g(a) > 0 (misclassified).
            # Constraint: g(a) + 1 <= y. If y=0 (correct), then g(a) <= -1.
            # So if g(a) > -1 (or > 0 broadly), it is NOT fully covered/inside.
            # Let's stick to the convention: Prune if 'covered'.
            # Covered means it satisfies the separation.
            # In PCF, we separate A from B.
            # g(x) calculated for A points.
            # If g(a_i) <= -1 (slack=0), it is covered.
            # We remove covered points.
            # So we KEEP points where g(a_i) > -1.
            # (Note: In RPCF code, we kept g(a) > 0. It's slightly looser. Let's start with > -1 or > 0).
            # RPCF code: keep_mask_A = g_vals_A > 0.
            # Let's stick to > 0 to be safe and consistent, or maybe > -0.99?
            # Let's use > 0 for strictness (meaning it's definitely not deep enough inside).

            g_vals_A = self._evaluate_g(
                X[A_indices], params["w"], params["xi"], params["gamma"], center_a
            )
            # We keep points that are NOT separated yet.
            # Separated/Covered means g(a) is negative (inside cone).
            keep_mask_A = (
                g_vals_A > 0.0
            )  # If positive, it's outside the cone (on the B side or boundary).
            A_indices = np.array(A_indices)[keep_mask_A].tolist()

            # B indices never change in PCF (Hard constraint ensures they assume correct side)
            # But just in case, we can check.
            pass

            # print(f"PCF Iter {iteration}: Remaining A: {len(A_indices)}")

    def _fit_ova(self, X, y):
        self.estimators_ = []
        # print(f"Multi-class PCF detected {self.classes_}. Training OvA...")
        for i, cls in enumerate(self.classes_):
            # print(f"  Training PCF Estimator for Class {cls} vs All...")
            y_binary = np.where(y == cls, -1, 1)
            estimator = copy.deepcopy(self)
            estimator.functions = []
            estimator.centers = []
            if hasattr(estimator, "classes_"):
                del estimator.classes_
            if hasattr(estimator, "estimators_"):
                del estimator.estimators_
            estimator.fit(X, y_binary)
            self.estimators_.append(estimator)

    def predict(self, X):
        if hasattr(self, "estimators_"):
            scores = self.decision_function(X)
            indices = np.argmax(scores, axis=1)
            return self.classes_[indices]
        else:
            g_min = self._compute_g_min(X)
            # g_min <= 0 -> Class -1 (Indies A)
            return np.where(g_min <= 0, -1, 1)

    def decision_function(self, X):
        if hasattr(self, "estimators_"):
            n_samples = X.shape[0]
            n_classes = len(self.classes_)
            scores = np.zeros((n_samples, n_classes))
            for i, estimator in enumerate(self.estimators_):
                g_vals = estimator._compute_g_min(X)
                scores[:, i] = -g_vals
            return scores
        else:
            return self._compute_g_min(X)

    def _compute_g_min(self, X):
        if not self.functions:
            return np.ones(len(X))
        g_matrix = np.zeros((len(X), len(self.functions)))
        for k, func in enumerate(self.functions):
            g_matrix[:, k] = self._evaluate_g(
                X, func["w"], func["xi"], func["gamma"], func["center"]
            )
        return np.min(g_matrix, axis=1)

    def select_center(self, candidates):
        return np.random.choice(candidates)
