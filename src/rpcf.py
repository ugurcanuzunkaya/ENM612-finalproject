import numpy as np
from src.solvers import solve_subproblem_qk


class RPCF:
    """
    Revisied - Çokyüzlü Konik Fonksiyonlar (r-PCF) Algoritması.

    Bu algoritma, -1 sınıfını (A Kümesi) +1 sınıfından (B Kümesi) ayırmak için
    yinelemeli olarak çokyüzlü konik fonksiyonlar ekleyerek bir sınıflandırma modeli oluşturur.
    Her iterasyonda A'dan doğru sınıflandırılan noktaların A boşalana (veya maksimum
    iterasyona ulaşılana) kadar çıkarıldığı bir "kurabiye kalıbı" yaklaşımı kullanır.
    """

    def __init__(self, C=1.0, lamb=0.01):
        self.C = C
        self.lamb = lamb
        self.functions = []  # Öğrenilen konik fonksiyonların listesi
        self.centers = []
        self.A_full = None
        self.B_full = None

    def _evaluate_g(self, X, w, xi, gamma, center):
        """
        Konik fonksiyon g(x)'in değerini hesaplar.
        g(x) = w'(x-a) + xi*||x-a||_1 - gamma
        """
        diff = X - center
        term1 = np.dot(diff, w)
        term2 = xi * np.sum(np.abs(diff), axis=1)
        return term1 + term2 - gamma

    def fit(self, X, y):
        # A (Sınıf -1) ve B'ye (Sınıf 1) ayır
        # İndeksleri TAM X'e göre saklıyoruz
        A_indices = np.where(y == -1)[0].tolist()
        B_indices = np.where(y == 1)[0].tolist()

        self.A_full = X
        self.B_full = X

        # A sınıfını B sınıfından yinelemeli olarak ayır.
        # İdeal olarak, kesişimi B'yi doğru sınıflandıran bir koni kümesi bulmak istiyoruz
        # ve A'yı hariç tutan. Yapıcı yaklaşımda, her adımda "kesilen"
        # (mevcut koninin dışında olarak doğru sınıflandırılan) noktaları A'dan çıkarıyoruz.

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

            # Modeli Sakla
            model_dict = {**params, "center": center_a}
            self.functions.append(model_dict)
            self.centers.append(center_a)

            # Veri setlerini budamak için değerlendir
            # A için: g(a) > 0 olan noktaları tut (Yanlış sınıflandırılmış/Kapsanmamış)
            g_vals_A = self._evaluate_g(
                X[A_indices], params["w"], params["xi"], params["gamma"], center_a
            )
            keep_mask_A = g_vals_A > 0
            A_indices = np.array(A_indices)[keep_mask_A].tolist()

            # B için: g(b) > 0 olan noktaları tut (Doğru sınıflandırılmış)
            g_vals_B = self._evaluate_g(
                X[B_indices], params["w"], params["xi"], params["gamma"], center_a
            )
            keep_mask_B = g_vals_B > 0
            B_indices = np.array(B_indices)[keep_mask_B].tolist()

            print(
                f"Iter {iteration}: Remaining A: {len(A_indices)}, B: {len(B_indices)}"
            )

    def select_center(self, candidates):
        # Varsayılan r-PCF: Rastgele seçim
        return np.random.choice(candidates)

    def predict(self, X):
        if not self.functions:
            return np.zeros(len(X))

        # g(x) = min(g_1, g_2, ... g_k)
        # min(g) <= 0 ise -1 olarak sınıflandır, aksi takdirde 1
        g_matrix = np.zeros((len(X), len(self.functions)))

        for k, func in enumerate(self.functions):
            g_matrix[:, k] = self._evaluate_g(
                X, func["w"], func["xi"], func["gamma"], func["center"]
            )

        g_min = np.min(g_matrix, axis=1)

        if not self.functions:
            return np.zeros(
                len(X)
            )  # Muhtemelen varsayılan bir sınıf döndürmeli veya bunu ele almalı.

        return np.where(g_min <= 0, -1, 1)
