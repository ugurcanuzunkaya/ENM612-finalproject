# r-PCF ve VNS-RPCF

Bu proje, **revisied Polyhedral Conic Functions (r-PCF)** algoritmasını ve performansını artırmak için geliştirilen **Değişken Komşuluk Arama (VNS)** tabanlı sezgisel yaklaşımı **(VNS-RPCF)** sunmaktadır. Çalışma, bu algoritmaları standart **PCF** ile karşılaştırmalı olarak analiz eder.

## Özet

Sınıflandırma problemlerinde, veriyi ayıran en iyi hiper düzlemlerin bulunması kritiktir. Mevcut PCF yöntemleri aşırı öğrenmeye (overfitting) yatkın olabilir. Bu çalışmada, **$l_2$-norm düzenlileştirmesi** eklenmiş yeni bir formülasyon olan **r-PCF** ve bu modelin hiperparametre ile küme merkezlerini optimize eden **VNS-RPCF** algoritması incelenmiştir. 11 farklı veri seti üzerinde yapılan kapsamlı deneyler, VNS-RPCF'in özellikle karmaşık ve doğrusal olmayan veri setlerinde standart r-PCF'e göre istatistiksel olarak anlamlı bir performans artışı sağladığını göstermektedir. Ayrıca, algoritmaların yorumlanabilirliği (daha az sayıda fonksiyon/merkez kullanımı) optimize edilmiştir.

---

## Metodoloji

Proje kapsamında üç temel algoritma karşılaştırılmıştır:

1. **r-PCF (revised Polyhedral Conic Functions):**
    * Orijinal PCF'ye $C$ (ceza parametresi) ve $\lambda$ (düzenlileştirme parametresi) eklenerek genelleştirilmiştir.
    * Amaç: Aşırı öğrenmeyi önlemek ve daha iyi genelleme yapmak.
    * Her iterasyonda karesel programlama (QP) alt problemi çözer.

2. **VNS-RPCF (Variable Neighborhood Search enhanced revised Polyhedral Conic Functions):**
    * r-PCF'in yerel minimumlara takılmasını önlemek için geliştirilmiştir.
    * Deterministik r-PCF çözümünü, rastgele yer değiştirmeler (shaking) ve yerel aramalar ile iyileştirir.
    * Daha kararlı ve yüksek doğruluklu modeller üretir.
    * Kullanılan operatörün görevi, r-PCF algoritmasının yerel minimumlara takılmasını önlemektir. Bunu da seçilen küme merkezlerinin yerini değiştirerek yapar. Bu sayede daha iyi performans elde edilir.
    * Destroy Operatörü: Seçilen küme merkezlerinin yerini değiştirir.
    * Repair Operatörü: Seçilen küme merkezlerini r-PCF algoritması ile iyileştirir.

3. **Standart PCF (Polyhedral Conic Functions):**
    * Düzenlileştirme içermeyen temel algoritma. Referans noktası olarak kullanılmıştır.
    * Her iterasyonda karesel programlama (QP) alt problemi çözer. Katı kısıtlamaları (hard constraints) vardır.

---

## Deneysel Kurulum

* **Veri Setleri:** UCI Makine Öğrenmesi deposundan ve sentetik kaynaklardan seçilen 11 veri seti.
* **Doğrulama Yöntemi:**
  * Küçük veri setleri ($N < 1000$): **10-Katmanlı Çapraz Doğrulama (10-Fold CV)**
  * Büyük veri setleri ($N \ge 1000$): **Hold-out (%80 Eğitim, %20 Test)**
* **Tekrarlar:** Deneyler **5 farklı rastgele tohum (seed)** ile tekrarlanmıştır.
* **Hiperparametre Optimizasyonu:**
  * Grid Search: $C$ ve $\lambda$ parametreleri için.
  * VNS: $k_{neighbors}$ (komşuluk sayısı) optimize edilmiştir.

---

## Deneysel Sonuçlar

### Tablo 1: Test Doğruluğu Performans Özeti (Ortalama ± Standart Sapma)

| Veri Seti | r-PCF | VNS-RPCF | PCF | En İyi Model |
| :--- | :--- | :--- | :--- | :--- |
| **Moons** | 0.9710 ± 0.036 | **0.9840 ± 0.029** | 0.9640 ± 0.038 | VNS-RPCF |
| **Ionosphere** | **0.9066 ± 0.061** | 0.9060 ± 0.046 | 0.8672 ± 0.053 | r-PCF / VNS |
| **Spambase** | 0.9216 ± 0.010 | **0.9231 ± 0.011** | 0.9006 ± 0.013 | VNS-RPCF |
| **WBCD** | 0.9698 ± 0.025 | 0.9698 ± 0.025 | 0.9396 ± 0.034 | r-PCF / VNS |
| **WBCP** | 0.7678 ± 0.094 | **0.7748 ± 0.076** | 0.7343 ± 0.079 | VNS-RPCF |
| **Votes** | **0.9425 ± 0.037** | 0.9411 ± 0.035 | 0.9114 ± 0.085 | r-PCF |
| **Breast Cancer** | 0.9607 ± 0.027 | **0.9625 ± 0.027** | 0.9649 ± 0.034 | PCF / VNS |
| **Blobs 3D** | **1.0000 ± 0.000** | **1.0000 ± 0.000** | **1.0000 ± 0.000** | Hepsi |
| **Liver** | **0.9445 ± 0.040** | 0.9442 ± 0.039 | 0.9252 ± 0.041 | r-PCF |
| **Heart** | **0.7547 ± 0.071** | 0.7492 ± 0.070 | 0.7477 ± 0.070 | r-PCF |
| **Statlog Heart** | 0.7422 ± 0.067 | 0.7511 ± 0.087 | **0.7622 ± 0.065** | PCF |

### Tablo 2: Eğitim Performansı ve Model Karmaşıklığı (Tipik Sonuçlar)

Bu tablo, algoritmaların eğitim başarısını, işlem yükünü (süre) ve model karmaşıklığını (merkez sayısı) göstermektedir.

| Veri Seti | Algoritma | Eğitim Doğruluğu (Train Acc) | Eğitim Süresi (s) | Merkez Sayısı (Centers) |
| :--- | :--- | :--- | :--- | :--- |
| **Moons** | r-PCF | 0.9852 | 0.023 | 5 |
| | VNS-RPCF | 0.9862 | 0.297 | 4.5 |
| | PCF | 1.0000 | 0.015 | 7.5 |
| **Ionosphere** | r-PCF | 0.9968 | 0.280 | 9 |
| | VNS-RPCF | 1.0000 | 3.758 | 11 |
| **Spambase** | r-PCF | 0.9867 | 23.34 | 216 |
| | VNS-RPCF | 0.9870 | 272.1 | 171 |
| **WBCD** | r-PCF | 0.9708 | 0.070 | 1 |
| | VNS-RPCF | 0.9727 | 1.330 | 1 |
| **Votes** | r-PCF | 0.9719 | 0.109 | 8 |
| | VNS-RPCF | 0.9719 | 1.516 | 9 |
| **Breast Cancer** | r-PCF | 0.9922 | 2.404 | 3 |
| | VNS-RPCF | 1.0000 | 0.019 (Tahmini) | 1 |
| **Heart** | r-PCF | 1.0000 | 0.490 | 44 |
| | VNS-RPCF | 1.0000 | 7.185 | 43 |
| **Blobs 3D** | r-PCF | 1.0000 | 0.009 | 1 |
| | VNS-RPCF | 1.0000 | 0.300 | 1 |

*Not: Süreler donanıma göre değişiklik gösterebilir ancak algoritmalar arası oransal farklar (VNS'in ek maliyeti vb.) tutarlıdır.*

### Temel Bulgular

1. **VNS Etkisi:** Karmaşık ve gürültülü veri setlerinde (örneğin **Moons**, **Spambase**, **WBCP**), VNS-RPCF en iyi test doğruluğunu elde etmiştir. Bu, yerel aramanın genelleme yeteneğini artırdığını doğrular.
2. **Eğitim Maliyeti:** VNS-RPCF, standart r-PCF'ye göre daha uzun eğitim sürelerine sahiptir (Tablo 2). Ancak bu maliyet, daha yüksek doğruluk ve genellikle daha az veya benzer sayıda merkez (daha sade model) ile telafi edilmektedir.
3. **Düzenlileştirme:** r-PCF serisi, düzenlileştirilmemiş PCF'ye göre çoğu durumda daha kararlıdır.

---

## Kurulum ve Kullanım

Proje, modern Python paket yöneticisi `uv` ile yapılandırılmıştır.

### Gereksinimler

* Python 3.10+
* `uv` (Önerilen) veya `pip`

### Çalıştırma

Proje artık modüler bir yapıya sahiptir. İki şekilde çalıştırılabilir:

#### 1. Tam Otomatik (Tek Komut)

Tüm süreci (Grid Search + Benchmark) sırayla çalıştırır.

```bash
# Varsayılan tüm veri setleri için:
uv run main.py

# Belirli veri setleri için:
uv run main.py moons heart
```

#### 2. Adım Adım (Modüler)

Süreci parçalara bölerek yönetebilirsiniz.

**Adım 1: Hiperparametre Optimizasyonu (Grid Search)**
En iyi parametreleri (C, lambda, k) bulur ve `solutions/grid_search/` altına `.json` olarak kaydeder.

```bash
uv run run_grid_search.py moons
```

**Adım 2: Benchmark Testleri**
Kaydedilen parametreleri okuyarak modelleri (r-PCF, VNS-RPCF, PCF) eğitir ve test eder.

```bash
uv run run_benchmark.py moons
```

**Ek Analizler:**

```bash
# Duyarlılık analizi
uv run python -m src.sensitivity moons
```

### Sonuçların İncelenmesi

Tüm çıktılar `solutions/` klasöründe düzenli bir yapıda saklanır:

* **`solutions/benchmarks/`**:
  * `*_results.txt`: Özet rapor (En iyi parametreler, istatistikler, doğruluk skorları).
  * `*_detailed_results.txt`: Detaylı rapor (Fonksiyon/merkez parametrelerini de içerir).
  * `*.png`: Karar sınırı grafikleri ve VNS yakınsama grafikleri.
* **`solutions/grid_search/`**:
  * `*_best_params.json`: Bulunan optimal parametreler (C, lambda, k).
  * `*_grid_search.txt`: Grid search detayları.
* **`solutions/stats/`**: İstatistiksel test sonuçları.
* **`solutions/sensitivity/`**: Duyarlılık analizi raporları.

---

## Lisans

Bu proje akademik araştırma amaçlı geliştirilmiştir. Tüm hakları saklıdır.
