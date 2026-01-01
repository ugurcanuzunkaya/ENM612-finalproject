from src.rpcf import RPCF
from src.solvers import solve_subproblem_qk
from sklearn.neighbors import NearestNeighbors
import numpy as np


class VNS_RPCF(RPCF):
    def __init__(
        self, C=1.0, lamb=0.01, k_neighbors=10, max_vns_iter=5, max_neighbors_check=5
    ):
        super().__init__(C, lamb)
        self.k_neighbors = k_neighbors
        self.max_vns_iter = max_vns_iter
        self.max_neighbors_check = max_neighbors_check

    def select_center(self, candidates_indices):
        """
        Overrides the random selection with VNS.
        We want to find a center 'a' that maximizes separation efficiency.
        """
        # candidates_indices is a list of valid indices in self.A_full

        # 1. Start with a random candidate
        current_best_idx = np.random.choice(candidates_indices)
        current_best_score = -np.inf

        # Build NN for local search space on the CURRENT candidates
        candidate_data = self.A_full[candidates_indices]

        # Safety check depending on number of candidates
        curr_k = min(self.k_neighbors, len(candidates_indices))
        if curr_k < 1:
            return current_best_idx

        nbrs_model = NearestNeighbors(n_neighbors=curr_k).fit(candidate_data)

        # We need to access B to evaluate performance
        current_B_indices = getattr(self, "current_B_indices", [])

        # Heuristic loop
        for vns_step in range(self.max_vns_iter):
            try:
                # Find position in candidates_indices list
                internal_idx = candidates_indices.index(current_best_idx)
            except ValueError:
                break

            # Get neighbors (indices in candidate_data)
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

                # Verify if we should test this neighbor (skip if same as current)
                if n_full_idx == current_best_idx and vns_step > 0:
                    continue

                # Solve QP
                center_candidate = self.A_full[n_full_idx]

                # To save time, we might sample B? Project Requirement doesn't mention sampling.
                params = solve_subproblem_qk(
                    candidates_indices,
                    current_B_indices,
                    self.A_full,
                    self.B_full,
                    center_candidate,
                    self.C,
                    self.lamb,
                )

                if params is None:
                    continue

                # Calculate Efficiency (Cut Volume)
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

            if not improved:
                # Shaking: Jump to a random other candidate
                idx_rand = np.random.choice(len(candidates_indices))
                current_best_idx = candidates_indices[idx_rand]

        return current_best_idx
