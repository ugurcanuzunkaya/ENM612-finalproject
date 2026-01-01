import gurobipy as gp
from gurobipy import GRB
import numpy as np


def solve_subproblem_qk(A_indices, B_indices, A_full, B_full, center_a, C, lamb):
    """
    Solves the QP subproblem for a given center.

    Args:
        A_indices: Current active indices for Set A (Class -1)
        B_indices: Current active indices for Set B (Class +1)
        A_full: Full dataset A (Class -1)
        B_full: Full dataset B (Class +1)
        center_a: The chosen center point (from A)
        C: Hyperparameter for misclassification penalty
        lamb: Hyperparameter for regularization

    Returns:
        Dictionary with optimal parameters w, xi, gamma, obj, or None if failed.
    """
    m_sub = len(A_indices)
    p_sub = len(B_indices)
    # If A is empty, we stop (Set A is fully covered).
    if m_sub == 0:
        return None

    n_features = A_full.shape[1]

    try:
        model = gp.Model("Q_k")
        model.setParam("OutputFlag", 0)

        # Variables
        w = model.addVars(n_features, lb=-GRB.INFINITY, name="w")
        xi = model.addVar(lb=0.0, name="xi")
        gamma = model.addVar(lb=1.0, name="gamma")
        y_slack = model.addVars(m_sub, lb=0.0, name="y")

        if p_sub > 0:
            z_slack = model.addVars(p_sub, lb=0.0, name="z")

        # Constraint: g(x) + 1 <= y_i
        for idx_enum, original_idx in enumerate(A_indices):
            point = A_full[original_idx]
            diff = point - center_a

            term1 = gp.LinExpr()
            term1.addTerms(diff, [w[j] for j in range(n_features)])

            l1_norm = np.sum(np.abs(diff))
            model.addConstr(term1 + l1_norm * xi - gamma + 1 <= y_slack[idx_enum])

        # Constraint: -g(x) + 1 <= z_j (Only if B exists)
        if p_sub > 0:
            for idx_enum, original_idx in enumerate(B_indices):
                point = B_full[original_idx]
                diff = point - center_a

                term1 = gp.LinExpr()
                term1.addTerms(diff, [w[j] for j in range(n_features)])

                l1_norm = np.sum(np.abs(diff))
                model.addConstr(
                    -1 * term1 - l1_norm * xi + gamma + 1 <= z_slack[idx_enum]
                )

        # Objective: lambda * (||w||^2 + xi^2 + gamma^2) + sum(y) + C * sum(z)
        w_sq = gp.quicksum(w[j] * w[j] for j in range(n_features))
        reg_term = w_sq + xi * xi + gamma * gamma

        # Normalization Weights (1/m for A, C/p for B)
        weight_A = 1.0 / m_sub if m_sub > 0 else 0.0
        weight_B = C / p_sub if p_sub > 0 else 0.0

        term_A = weight_A * gp.quicksum(y_slack)
        term_B = weight_B * gp.quicksum(z_slack) if p_sub > 0 else 0.0

        obj = lamb * reg_term + term_A + term_B
        model.setObjective(obj, GRB.MINIMIZE)

        model.optimize()

        if model.status == GRB.OPTIMAL:
            return {
                "w": np.array([w[j].X for j in range(n_features)]),
                "xi": xi.X,
                "gamma": gamma.X,
                "obj": model.ObjVal,
            }
        else:
            return None

    except gp.GurobiError as e:
        print(f"Gurobi Error: {e}")
        return None
