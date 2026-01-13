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
        self.all_vns_histories = []  # Stores score history for each call to select_center

    def select_center(self, candidates_indices):
        """
        Değişken Komşuluk Araması (VNS) kullanarak en iyi merkezi seçer.
        """
        # candidates_indices, self.A_full içindeki geçerli indekslerin bir listesidir

        # 1. Rastgele bir aday ile başla
        current_best_idx = np.random.choice(candidates_indices)
        current_best_score = -np.inf

        step_history = []

        # MEVCUT adaylar üzerinde yerel arama uzayı için NN oluştur
        candidate_data = self.A_full[candidates_indices]

        # Aday sayısına bağlı güvenlik kontrolü
        curr_k = min(self.k_neighbors, len(candidates_indices))
        if curr_k < 1:
            self.all_vns_histories.append([0])
            return current_best_idx

        nbrs_model = NearestNeighbors(n_neighbors=curr_k).fit(candidate_data)

        # Performansı değerlendirmek için B'ye erişmemiz gerekiyor
        current_B_indices = getattr(self, "current_B_indices", [])

        # Sezgisel döngü
        for vns_step in range(self.max_vns_iter):
            # Record current best score at start of step (or end?)
            # Initial score calculation is expensive if we haven't computed it yet.
            # But we only need improvement. Let's record *updates*.
            # Actually, let's try to estimate or just record when it improves.
            # Or better: initialize current_best_score properly if possible, but random choice has unknown score.
            # Let's assume score starts low.

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

            # Record history after looking at neighbors
            # Note: If current_best_score is -inf (no update), we might want to record 0 or skip
            # But if it improved, we record the new score.
            # If it didn't improve, the score remains same (or -inf if never found valid).
            valid_score = current_best_score if current_best_score != -np.inf else 0
            step_history.append(valid_score)

            if not improved:
                # Çalkalama (Shaking): Rastgele başka bir adaya atla
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
