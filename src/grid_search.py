from sklearn.metrics import accuracy_score
from src.rpcf import RPCF


def grid_search_rpcf(X_train, y_train, X_val, y_val):
    """
    Doğrulama seti üzerinde r-PCF modeli için en iyi hiperparametreleri (C, lambda)
    bulmak amacıyla basit bir ızgara araması (grid search) gerçekleştirir.

    Args:
        X_train, y_train: Eğitim verisi
        X_val, y_val: Doğrulama verisi

    Returns:
        dict: Bulunan en iyi 'C' ve 'lamb' değerlerini içeren bir sözlük.
    """
    best_acc = -1.0
    best_params = {"C": 1.0, "lamb": 0.01}

    # Makalede önerilen aralık (hız için basitleştirildi)
    C_values = [0.1, 1, 10, 100]
    lamb_values = [0.01, 0.1, 1]

    curr = 0

    for C in C_values:
        for lamb in lamb_values:
            curr += 1

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
