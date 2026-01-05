"""
Veri Yükleme Modülü.

Bu modül, sklearn ve UCI Makine Öğrenimi Deposu'ndan çeşitli veri setlerini
getirmek, yüklemek ve ön işlemekle sorumludur. Normalizasyon (StandardScaler)
ve r-PCF metniyle uyumluluğu sağlamak için etiket kodlamasını yönetir.
"""

import numpy as np
from sklearn.datasets import make_moons, load_breast_cancer, make_blobs
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

try:
    from ucimlrepo import fetch_ucirepo
except ImportError:
    fetch_ucirepo = None


class DatasetLoader:
    """
    Veri setlerini yüklemek ve ön işlemek için merkezi işleyici.
    Hem sentetik (sklearn) hem de gerçek dünya (UCI) veri setlerini destekler.
    """

    def __init__(self):
        self.scaler = StandardScaler()

    def load_dataset(self, dataset_name):
        """
        İsme göre belirli veri seti yükleyicisine yönlendirir.
        """
        if dataset_name == "moons":
            return self.load_moons()
        elif dataset_name == "breast_cancer":
            return self.load_breast_cancer()
        elif dataset_name == "blobs_3d":
            return self.load_blobs_3d()
        elif dataset_name == "wbcd":
            return self.load_wbcd()
        elif dataset_name == "wbcp":
            return self.load_wbcp()
        elif dataset_name == "heart":
            return self.load_heart()
        elif dataset_name == "liver":
            return self.load_liver()
        elif dataset_name == "votes":
            return self.load_votes()
        elif dataset_name == "ionosphere":
            return self.load_ionosphere()
        else:
            raise ValueError(
                f"Dataset '{dataset_name}' not found. Available: moons, breast_cancer, blobs_3d, wbcd, wbcp, heart, liver, votes, ionosphere"
            )

    def load_moons(self):
        print("\n--- Loading 'Moons' Dataset ---")
        X, y = make_moons(n_samples=200, noise=0.15, random_state=42)
        X = self.scaler.fit_transform(X)
        return X, y

    def load_breast_cancer(self):
        print("\n--- Loading 'Breast Cancer' Dataset ---")
        data = load_breast_cancer()
        X, y = data.data, data.target
        X = self.scaler.fit_transform(X)
        return X, y

    def load_blobs_3d(self):
        print("\n--- Loading 'Blobs 3D' Dataset ---")
        X, y = make_blobs(n_samples=200, centers=2, n_features=3, random_state=42)
        X = self.scaler.fit_transform(X)
        return X, y

    def load_wbcd(self):
        """
        Wisconsin Meme Kanseri Teşhisi (WBCD) veri setini yükler.
        UCI Repo ID: 17
        Hedef: Teşhis (M = kötü huylu, B = iyi huylu)
        """
        print("\n--- Loading 'WBCD' (Diagnosis) Dataset [via ucimlrepo] ---")
        if fetch_ucirepo is None:
            raise ImportError("ucimlrepo not installed")

        # Veri setini getir
        dataset = fetch_ucirepo(id=17)

        # Özellikleri ve hedefleri çıkar
        X = dataset.data.features
        y = dataset.data.targets

        # Numpy dizilerine dönüştür
        X = X.values
        y = y.values.ravel()  # (n_samples,) şekline düzleştir

        # Hedefi Kodla (M/B -> 1/0)
        le = LabelEncoder()
        y = le.fit_transform(y)

        # Ölçeklendirme
        if hasattr(self, "scaler") and self.scaler:
            X = self.scaler.fit_transform(X)

        return X, y

    def load_wbcp(self):
        """
        Wisconsin Meme Kanseri Prognozu (WBCP) veri setini yükler.
        UCI Repo ID: 16
        Hedef: Sonuç (R = nükseden, N = nüksetmeyen)
        """
        print("\n--- Loading 'WBCP' (Prognosis) Dataset [via ucimlrepo] ---")
        if fetch_ucirepo is None:
            raise ImportError("ucimlrepo not installed")

        # Veri setini getir
        dataset = fetch_ucirepo(id=16)

        X = dataset.data.features
        y = dataset.data.targets

        # Ön işleme
        X = X.values
        y = y.values.ravel()

        # Sadece özellikleri istiyorsanız 'Zaman' sütununu atın (isteğe bağlı, hedefe bağlı)
        # Genellikle 'Zaman', nüksün saf sınıflandırması için hariç tutulur
        # X = X[:, 1:]

        # Hedefi Kodla (R/N -> 1/0)
        le = LabelEncoder()
        y = le.fit_transform(y)

        # Eğer varsa NaN değerleri doldur (WBCP 'Lenf nodu durumu' vb. yerlerde eksik değerlere sahiptir)
        imputer = SimpleImputer(strategy="mean")
        X = imputer.fit_transform(X)

        # Ölçeklendirme
        if hasattr(self, "scaler") and self.scaler:
            X = self.scaler.fit_transform(X)

        return X, y

    def load_heart(self):
        """
        Cleveland Kalp Hastalığı veri setini yükler.
        UCI Repo ID: 45
        Hedef: Teşhis (0=sağlıklı, 1-4=hasta)
        """
        print("\n--- Loading 'Cleveland Heart' Dataset [via ucimlrepo] ---")
        if fetch_ucirepo is None:
            raise ImportError("ucimlrepo not installed")

        # Veri setini getir (ID 45 ana Kalp Hastalığı konteyneridir)
        dataset = fetch_ucirepo(id=45)

        X = dataset.data.features
        y = dataset.data.targets

        # Numpy'ye dönüştür
        X = X.values
        y = y.values.ravel()

        # Eksik değerleri doldur (Yeni repoda NaN olabilir)
        imputer = SimpleImputer(strategy="mean")
        X = imputer.fit_transform(X)

        # Hedefi İkili Hale Getir: 0 sağlıklı, >0 kalp hastalığı
        y = np.where(y > 0, 1, 0)

        # Ölçeklendirme
        if hasattr(self, "scaler") and self.scaler:
            X = self.scaler.fit_transform(X)

        return X, y

    def load_liver(self):
        """
        BUPA Karaciğer Bozuklukları veri setini yükler.
        UCI Repo ID: 60
        """
        print("\n--- Loading 'BUPA Liver' Dataset [via ucimlrepo] ---")
        if fetch_ucirepo is None:
            raise ImportError("ucimlrepo not installed")

        dataset = fetch_ucirepo(id=60)

        X = dataset.data.features
        y = dataset.data.targets

        X = X.values
        y = y.values.ravel()

        # BUPA'daki hedef genellikle 'seçici'dir (alan 7).
        # 0/1 olarak kodlandığından emin olun (başlangıçta 1/2 olabilir).
        le = LabelEncoder()
        y = le.fit_transform(y)

        # Ölçeklendirme
        if hasattr(self, "scaler") and self.scaler:
            X = self.scaler.fit_transform(X)

        return X, y

    def load_votes(self):
        """
        A.B.D. Kongre Oylama Kayıtları veri setini yükler.
        UCI Repo ID: 105
        Hedef: Parti (Demokrat/Cumhuriyetçi)
        """
        print("\n--- Loading 'Congress Voting' Dataset [via ucimlrepo] ---")
        if fetch_ucirepo is None:
            raise ImportError("ucimlrepo not installed")

        dataset = fetch_ucirepo(id=105)

        X = dataset.data.features
        y = dataset.data.targets

        # Özelliklerdeki '?' veya NaN'leri ele alma (Oylama kayıtlarında çok fazla çekimser oy vardır)
        # ucimlrepo bunları genellikle NaN olarak yükler.
        # Strateji: 'en sık' (most_frequent) ile doldurun veya ayrı bir kategori olarak ele alın.

        # Eğer dize iseler 'y'/'n' değerlerini manuel olarak 1/0'a eşleyin
        if hasattr(X, "replace"):
            X = X.replace({"y": 1, "n": 0, "?": np.nan})

        imputer = SimpleImputer(strategy="most_frequent")
        X = imputer.fit_transform(X)

        y = y.values.ravel()

        # Hedefi Kodla (demokrat/cumhuriyetçi -> 0/1)
        le = LabelEncoder()
        y = le.fit_transform(y)

        # Ölçeklendirme
        if hasattr(self, "scaler") and self.scaler:
            X = self.scaler.fit_transform(X)

        return X, y

    def load_ionosphere(self):
        """
        İyonosfer veri setini yükler.
        UCI Repo ID: 52
        Hedef: Sınıf (g=iyi, b=kötü)
        """
        print("\n--- Loading 'Ionosphere' Dataset [via ucimlrepo] ---")
        if fetch_ucirepo is None:
            raise ImportError("ucimlrepo not installed")

        dataset = fetch_ucirepo(id=52)

        X = dataset.data.features
        y = dataset.data.targets

        X = X.values
        y = y.values.ravel()

        # Hedefi Kodla
        le = LabelEncoder()
        y = le.fit_transform(y)

        # Ölçeklendirme
        if hasattr(self, "scaler") and self.scaler:
            X = self.scaler.fit_transform(X)

        return X, y

    def load_custom_dataset(self):
        """
        ŞABLON: Kendi veri setinizi eklemek için bu yöntemi kullanın.

        Adımlar:
        1. Verinizi yükleyin (ör. CSV, Excel veya kütüphaneden).
        2. Özellikleri (X) ve hedefi (y) ayırın.
        3. X'in (n_samples, n_features) şeklinde bir numpy dizisi olduğundan emin olun.
        4. y'nin (n_samples,) şeklinde bir numpy dizisi olduğundan emin olun.
        5. Gerekirse ölçeklendirme uygulayın.
        6. X, y değerlerini döndürün.
        """
        print("\n--- Loading 'Custom' Dataset ---")
        # --- YOUR CODE HERE ---
        # Example:
        # data = pd.read_csv("my_data.csv")
        # X = data.drop("target", axis=1).values
        # y = data["target"].values

        # Yer tutucu (Bunu verinizle değiştirin)
        X = np.random.randn(100, 5)  # 100 örnek, 5 özellik
        y = np.random.randint(0, 2, 100)  # İkili hedef

        # Ölçeklendirme
        X = self.scaler.fit_transform(X)

        return X, y
