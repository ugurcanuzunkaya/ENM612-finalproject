from src.rpcf import RPCF
from src.solvers import solve_subproblem_qk
from sklearn.neighbors import NearestNeighbors
import numpy as np


class VNS_RPCF(RPCF):
    """
    VNS-RPCF: Değişken Komşuluk Araması (Variable Neighborhood Search) ile geliştirilmiş r-PCF.

    Bu sınıf, rastgele merkez seçim stratejisini bir meta-sezgisel arama (VNS) ile
    değiştirerek minimal r-PCF algoritmasını genişletir. Her iterasyonda ayırma
    verimliliğini (çıkarılan A hacmi) maksimize eden optimal bir 'a' merkezi
    bulmaya çalışır.
    """

    def __init__(
        self, C=1.0, lamb=0.01, k_neighbors=10, max_vns_iter=5, max_neighbors_check=5
    ):
        super().__init__(C, lamb)
        self.k_neighbors = k_neighbors
        self.max_vns_iter = max_vns_iter
        self.max_neighbors_check = max_neighbors_check

    def select_center(self, candidates_indices):
        """
        Değişken Komşuluk Araması (VNS) kullanarak en iyi merkezi seçer.
        """
        # candidates_indices, self.A_full içindeki geçerli indekslerin bir listesidir

        # 1. Rastgele bir aday ile başla
        current_best_idx = np.random.choice(candidates_indices)
        current_best_score = -np.inf

        # MEVCUT adaylar üzerinde yerel arama uzayı için NN oluştur
        candidate_data = self.A_full[candidates_indices]

        # Aday sayısına bağlı güvenlik kontrolü
        curr_k = min(self.k_neighbors, len(candidates_indices))
        if curr_k < 1:
            return current_best_idx

        nbrs_model = NearestNeighbors(n_neighbors=curr_k).fit(candidate_data)

        # Performansı değerlendirmek için B'ye erişmemiz gerekiyor
        current_B_indices = getattr(self, "current_B_indices", [])

        # Sezgisel döngü
        for vns_step in range(self.max_vns_iter):
            try:
                # candidates_indices listesindeki pozisyonu bul
                internal_idx = candidates_indices.index(current_best_idx)
            except ValueError:
                break

            # Komşuları al (candidate_data içindeki indeksler)
            distances, indices = nbrs_model.kneighbors([candidate_data[internal_idx]])
            neighbor_internal_indices = indices[0]

            # Komşuları kontrol et
            improved = False
            checked_count = 0
            for n_int_idx in neighbor_internal_indices:
                if checked_count >= self.max_neighbors_check:
                    break
                checked_count += 1

                n_full_idx = candidates_indices[n_int_idx]

                # Bu komşuyu test edip etmeyeceğimizi doğrula (mevcut ile aynıysa atla)
                if n_full_idx == current_best_idx and vns_step > 0:
                    continue

                # QP'yi çöz
                center_candidate = self.A_full[n_full_idx]

                # QP'yi çöz
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

                # Verimliliği Hesapla (Kesilen Hacim)
                g_vals = self._evaluate_g(
                    self.A_full[candidates_indices],
                    params["w"],
                    params["xi"],
                    params["gamma"],
                    center_candidate,
                )

                # Doğru sınıflandırılmış A (çıkarılanlar), g(a) <= 0 olanlardır
                removed_count = np.sum(g_vals <= 0)
                score = removed_count

                if score > current_best_score:
                    current_best_score = score
                    current_best_idx = n_full_idx
                    improved = True
                    # İlk İyileştirme
                    break

            if not improved:
                # Çalkalama (Shaking): Rastgele başka bir adaya atla
                idx_rand = np.random.choice(len(candidates_indices))
                current_best_idx = candidates_indices[idx_rand]

        return current_best_idx
