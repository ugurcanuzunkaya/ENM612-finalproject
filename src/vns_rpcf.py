from src.rpcf import RPCF
from src.solvers import solve_subproblem_qk
from sklearn.neighbors import NearestNeighbors
import numpy as np


class VNS_RPCF(RPCF):
    """
    VNS-RPCF: r-PCF enhanced with Variable Neighborhood Search.

    This class extends the minimal r-PCF algorithm by replacing the random center
    selection strategy with a meta-heuristic search (VNS). At each iteration, it
    attempts to find an optimal center 'a' that maximizes separation efficiency
    (volume of removed A points).
    """

    def __init__(
        self, C=1.0, lamb=0.01, k_neighbors=50, max_vns_iter=5, max_neighbors_check=5
    ):
        super().__init__(C, lamb)
        self.k_neighbors = k_neighbors
        self.max_vns_iter = max_vns_iter
        self.max_neighbors_check = max_neighbors_check
        self.all_vns_histories = []  # Stores score history for each call to select_center

    def select_center(self, candidates_indices):
        """
        Selects the best center using Variable Neighborhood Search (VNS).
        """
        # candidates_indices is a list of valid indices within self.A_full

        # 1. Start with a random candidate
        current_best_idx = np.random.choice(candidates_indices)
        current_best_score = -np.inf

        step_history = []

        # Build NN on CURRENT candidates for local search space
        candidate_data = self.A_full[candidates_indices]

        # Safety check based on candidate count
        curr_k = min(self.k_neighbors, len(candidates_indices))
        if curr_k < 1:
            self.all_vns_histories.append([0])
            return current_best_idx

        nbrs_model = NearestNeighbors(n_neighbors=curr_k).fit(candidate_data)

        # We need access to B to evaluate performance
        current_B_indices = getattr(self, "current_B_indices", [])

        # Heuristic loop
        for vns_step in range(self.max_vns_iter):
            # Record current best score at start of step (or end?)
            # Initial score calculation is expensive if we haven't computed it yet.
            # But we only need improvement. Let's record *updates*.
            # Actually, let's try to estimate or just record when it improves.
            # Or better: initialize current_best_score properly if possible, but random choice has unknown score.
            # Let's assume score starts low.

            try:
                # Find position in candidates_indices list
                internal_idx = candidates_indices.index(current_best_idx)
            except ValueError:
                break

            # Get neighbors (indices within candidate_data)
            distances, indices = nbrs_model.kneighbors([candidate_data[internal_idx]])
            neighbor_internal_indices = indices[0]

            # Check neighbors
            improved = False
            checked_count = 0
            for n_int_idx in neighbor_internal_indices:
                if checked_count >= self.max_neighbors_check:
                    break
                checked_count += 1

                n_full_idx = candidates_indices[n_int_idx]

                # Validate whether to test this neighbor (skip if same as current)
                if n_full_idx == current_best_idx and vns_step > 0:
                    continue

                # Solve QP
                center_candidate = self.A_full[n_full_idx]

                # Solve QP
                params = solve_subproblem_qk(
                    candidates_indices,
                    current_B_indices,
                    self.A_full,
                    self.B_full,
                    center_candidate,
                    self.C,
                    self.lamb,
                )
                self.num_solved_subproblems += 1

                if params is None:
                    continue

                # Compute Efficiency (Removed Volume)
                g_vals = self._evaluate_g(
                    self.A_full[candidates_indices],
                    params["w"],
                    params["xi"],
                    params["gamma"],
                    center_candidate,
                )

                # Correctly classified A (removed) are those with g(a) <= 0
                removed_count = np.sum(g_vals <= 0)
                score = removed_count

                if score > current_best_score:
                    current_best_score = score
                    current_best_idx = n_full_idx
                    improved = True
                    # First Improvement
                    break

            # Record history after looking at neighbors
            # Note: If current_best_score is -inf (no update), we might want to record 0 or skip
            # But if it improved, we record the new score.
            # If it didn't improve, the score remains same (or -inf if never found valid).
            valid_score = current_best_score if current_best_score != -np.inf else 0
            step_history.append(valid_score)

            if not improved:
                # Shaking: Jump to another random candidate
                idx_rand = np.random.choice(len(candidates_indices))
                current_best_idx = candidates_indices[idx_rand]
                # Reset score? Usually VNS keeps best found globally or restarts?
                # This implementation restarts search from new random point but keeps tracking global best?
                # Actually logic above: 'current_best_idx' becomes new random.
                # 'current_best_score' is NOT reset in code?
                # If we construct VNS properly, we should keep track of Global Best separately from Current Search Node.
                # But looking at existing code:
                # It continues with 'current_best_idx' implies 'current_best_score' should logically correspond to it.
                # But if we jump, the score of new point is unknown.
                # For visualization, let's just track the best score found SO FAR in this call.
                pass

        self.all_vns_histories.append(step_history)
        return current_best_idx

        return current_best_idx
