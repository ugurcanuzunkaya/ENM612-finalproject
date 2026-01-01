from sklearn.metrics import accuracy_score
from src.rpcf import RPCF


def grid_search_rpcf(X_train, y_train, X_val, y_val):
    """
    Performs grid search to find best C and lambda parameters.
    """
    best_acc = -1.0
    best_params = {"C": 1.0, "lamb": 0.01}

    # Paper-suggested range (simplified for speed)
    C_values = [0.1, 1, 10, 100]
    lamb_values = [0.01, 0.1, 1]

    curr = 0

    for C in C_values:
        for lamb in lamb_values:
            curr += 1
            # print(f"  Grid Search {curr}/{total_combs}: C={C}, lamb={lamb}...", end="\r")

            try:
                model = RPCF(C=C, lamb=lamb)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                acc = accuracy_score(y_val, y_pred)

                if acc > best_acc:
                    best_acc = acc
                    best_params = {"C": C, "lamb": lamb}
            except Exception:
                continue

    print(f"  Best Grid Params: {best_params} (Acc: {best_acc:.4f})")
    return best_params
