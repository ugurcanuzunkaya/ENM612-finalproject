import gurobipy as gp
from gurobipy import GRB
import numpy as np


def solve_subproblem_qk(A_indices, B_indices, A_full, B_full, center_a, C, lamb):
    """
    Belirli bir merkez için QP alt problemini çözer.

    Args:
        A_indices: A Kümesi (Sınıf -1) için mevcut aktif indeksler
        B_indices: B Kümesi (Sınıf +1) için mevcut aktif indeksler
        A_full: Tam veri seti A (Class -1)
        B_full: Tam veri seti B (Class +1)
        center_a: Seçilen merkez noktası (A'dan)
        C: Hatalı sınıflandırma cezası için hiperparametre
        lamb: Düzenlileştirme (regularization) için hiperparametre

    Returns:
        Optimal parametreler w, xi, gamma, obj içeren sözlük veya başarısız olursa None.
    """
    m_sub = len(A_indices)
    p_sub = len(B_indices)
    # A boşsa dururuz (A Kümesi tamamen kapsanmıştır).
    if m_sub == 0:
        return None

    n_features = A_full.shape[1]

    try:
        model = gp.Model("Q_k")
        model.setParam("OutputFlag", 0)

        # Değişkenler
        w = model.addVars(n_features, lb=-GRB.INFINITY, name="w")
        xi = model.addVar(lb=0.0, name="xi")
        gamma = model.addVar(lb=1.0, name="gamma")
        y_slack = model.addVars(m_sub, lb=0.0, name="y")

        if p_sub > 0:
            z_slack = model.addVars(p_sub, lb=0.0, name="z")

        # Kısıt 1: A'daki x için g(x) >= 0 (slack ile formülasyonda kesinlikle > -1)
        # Resmi olarak: g(a_i) + 1 <= y_i  --> y_i > 0 ise yanlış sınıflandırılmış
        for idx_enum, original_idx in enumerate(A_indices):
            point = A_full[original_idx]
            diff = point - center_a

            term1 = gp.LinExpr()
            term1.addTerms(diff, [w[j] for j in range(n_features)])

            l1_norm = np.sum(np.abs(diff))
            model.addConstr(term1 + l1_norm * xi - gamma + 1 <= y_slack[idx_enum])

        # Kısıt 2: B'deki x için g(x) <= 0 (kesinlikle < 1)
        # Resmi olarak: -g(b_j) + 1 <= z_j --> z_j > 0 ise yanlış sınıflandırılmış
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

        # Amaç: Min lambda*(||w||^2 + xi^2 + gamma^2) + 1/m * sum(y) + C/p * sum(z)
        # Düzenlileştirme terimini ve ağırlıklı sınıflandırma hatalarını minimize ediyoruz.
        w_sq = gp.quicksum(w[j] * w[j] for j in range(n_features))
        reg_term = w_sq + xi * xi + gamma * gamma

        # Normalizasyon Ağırlıkları (A için 1/m, B için C/p)
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
