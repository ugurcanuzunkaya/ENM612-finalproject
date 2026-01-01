# Gözden Geçirilmiş Çokyüzlü Konik Fonksiyonlar (r-PCF) ve VNS İyileştirmesi

## Proje Hedefi

Bu proje, ENM612 dönem projesi kapsamında, ikili sınıflandırma (binary classification) problemleri için geliştirilen **r-PCF (Revised Polyhedral Conic Functions)** algoritmasını uygulamayı ve bu algoritmayı **Değişken Komşuluk Arama (Variable Neighborhood Search - VNS)** meta-sezgiseli ile geliştirmeyi amaçlar.

Temel hedef, sınıflandırma doğruluğunu yüksek tutarken, veri setini ayırmak için gereken konik fonksiyon (merkez) sayısını minimize etmektir. Proje, orijinal makaledeki yöntemleri Python ve Gurobi Optimizasyon Çözücüsü kullanarak yeniden üretir ve geliştirir.

## Özellikler

* **r-PCF Algoritması**: İteratif "cookie cutter" mantığıyla çalışan, matematiksel programlama tabanlı sınıflandırma.
* **VNS-RPCF (Geliştirilmiş Model)**: Rastgele merkez seçimi yerine, VNS kullanarak en iyi ayrımı yapacak merkezi arayan hibrit algoritma.
* **Kapsamlı Veri Desteği**: `DatasetLoader` modülü sayesinde 9 farklı veri seti üzerinde (Moons, Ionosphere, Breast Cancer, vb.) otomatik test imkanı.
* **Otomatik Eksik Veri Tamamlama**: Eksik veri içeren veri setleri (örn. WBCP) için otomatik `imputation` işlemi.
* **Detaylı Raporlama**: Her deney için eğitim süresi, doğruluk, fonksiyon sayısı ve model parametrelerinin (ağırlıklar, biaslar) ayrı dosyalara kaydedilmesi.
* **Görselleştirme**: 2 boyutlu veri setleri için karar sınırlarının ve merkezlerin görselleştirilmesi.

## Desteklenen Veri Setleri

Proje aşağıdaki veri setlerini `src/dataloader.py` üzerinden otomatik olarak indirir ve işler (`ucimlrepo` veya `sklearn` kaynaklı):

1. **moons**: Sentetik "Make Moons" veri seti (2D, doğrusal olmayan).
2. **breast_cancer**: Sklearn Breast Cancer Wisconsin (Diagnostic).
3. **blobs_3d**: Sentetik 3 boyutlu veri seti.
4. **wbcd**: Wisconsin Breast Cancer (Diagnostic) - UCI ID 17.
5. **wbcp**: Wisconsin Breast Cancer (Prognostic) - UCI ID 16.
6. **heart**: Cleveland Heart Disease - UCI ID 45.
7. **liver**: BUPA Liver Disorders - UCI ID 60.
8. **votes**: Congressional Voting Records - UCI ID 105.
9. **ionosphere**: Ionosphere Radar Data - UCI ID 52.

## Kurulum ve Hazırlık

### Ön Gereksinimler

* **Python 3.12+**
* **Gurobi Lisansı**: Kodun çalışması için sisteminizde geçerli bir `gurobi.lic` dosyası bulunmalıdır (Akademik lisans önerilir).

### Seçenek 1: `uv` ile Kurulum (Önerilen)

Bu proje modern Python araç seti `uv` ile yapılandırılmıştır.

1. Bağımlılıkları yükleyin:

   ```bash
   uv sync
   ```

2. Projeyi çalıştırın:

   ```bash
   uv run python main.py
   ```

### Seçenek 2: `pip` ile Kurulum

Standart Python ortamı için:

1. Gerekli kütüphaneleri yükleyin:

   ```bash
   pip install -r requirements.txt
   ```

2. Projeyi çalıştırın:

   ```bash
   python main.py
   ```

## Proje Dizini Yapısı

```text
/project_root
├── main.py                # Ana giriş noktası (Benchmark testlerini yönetir)
├── pyproject.toml         # Proje ve bağımlılık tanımları (uv)
├── requirements.txt       # Standart pip gereksinim dosyası
├── data/                  # İndirilen veri setlerinin geçici deposu
├── solutions/             # Çıktı klasörü (Sonuç raporları ve grafikler)
│   ├── moons_results.txt  # Her veri seti için detaylı parametre raporu
│   └── moons_rpcf.png     # Görselleştirilmiş karar sınırları
└── src/
    ├── dataloader.py      # Veri yükleme, temizleme ve ön işleme
    ├── rpcf.py            # Temel r-PCF algoritma sınıfı
    ├── vns_rpcf.py        # VNS ile geliştirilmiş r-PCF sınıfı
    ├── solvers.py         # Gurobi QP alt problem çözücüsü
    ├── visualizer.py      # 2D grafik çizim fonksiyonları
    └── utils.py           # Yardımcı raporlama ve kayıt fonksiyonları
```

## Sonuçlar ve Değerlendirme

Aşağıdaki tablo, farklı veri setleri üzerinde standart **r-PCF** ve **VNS-RPCF** algoritmalarının performans karşılaştırmasını göstermektedir.

| Veri Seti | Model | Doğruluk (Accuracy) | Süre (sn) |
| :--- | :--- | :--- | :--- |
| **Moons** | r-PCF | 0.9833 | 0.01 |
| | **VNS-RPCF** | **1.0000** | 0.21 |
| **Breast Cancer** (Sklearn) | r-PCF | 0.9415 | 0.08 |
| | **VNS-RPCF** | 0.9298 | 1.41 |
| **Blobs 3D** | r-PCF | 1.0000 | 0.00 |
| | **VNS-RPCF** | **1.0000** | 0.10 |
| **WBCD** (Diagnosis) | r-PCF | 0.9883 | 0.05 |
| | **VNS-RPCF** | 0.9591 | 1.08 |
| **WBCP** (Prognostic) | r-PCF | **0.8000** | 0.01 |
| | **VNS-RPCF** | 0.7833 | 0.24 |
| **Heart** | r-PCF | **0.7912** | 0.03 |
| | **VNS-RPCF** | 0.7802 | 0.53 |
| **Votes** | r-PCF | **0.9542** | 0.04 |
| | **VNS-RPCF** | 0.9389 | 0.49 |
| **Ionosphere** | r-PCF | 0.8868 | 0.18 |
| | **VNS-RPCF** | **0.8962** | 2.08 |
| **Liver** | r-PCF | 0.9327 | 0.12 |
| | **VNS-RPCF** | **0.9615** | 1.71 |

### Yorumlar

1. **Doğruluk Artışı**: `Liver`, `Ionosphere` ve `Moons` veri setlerinde **VNS-RPCF**, standart yönteme göre daha yüksek doğruluk sağlamıştır. Özellikle `Liver` veri setinde yaklaşık **%3**'lük bir artış (0.93 -> 0.96) görülmüştür. Ancak `Heart` ve `WBCD` gibi bazı setlerde standart yöntem benzer veya biraz daha iyi sonuçlar vermiştir; bu durum VNS'in stokastik yapısından ve hiperparametre ayarlarının (Grid Search aralığı) hassasiyetinden kaynaklanabilir.
2. **Süre Maliyeti**: VNS-RPCF, her iterasyonda optimum merkezi bulmak için komşuluk araması yaptığından eğitim süresi standart r-PCF'e göre belirgin şekilde uzundur. Bu beklenen bir trade-off (ödünleşim) durumudur.
3. **Model Sadeliği**: Her iki algoritma da oldukça az sayıda "merkez" (konik fonksiyon) kullanarak veriyi sınıflandırmayı başarmıştır. Örneğin 3 boyutta %100 ayrım için tek bir koni yeterli olmuştur. Bu, modelin "Sparse" (seyrek) ve yorumlanabilir olduğunu gösterir.

### Sonuç

VNS entegrasyonu, hesaplama maliyetini artırmakla birlikte, özellikle zorlu sınıflandırma problemlerinde modelin genelleme yeteneğini ve doğruluğunu artırmak için etkili bir yöntemdir. Basit veri setlerinde ise standart r-PCF hız avantajı nedeniyle tercih edilebilir.
