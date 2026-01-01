# Gözden Geçirilmiş Çokyüzlü Konik Fonksiyonlar (r-PCF) ve VNS İyileştirmesi

## Proje Hedefi

Bu proje, ENM612 dönem projesi kapsamında, ikili sınıflandırma (binary classification) problemleri için geliştirilen **r-PCF (Revised Polyhedral Conic Functions)** algoritmasını uygulamayı ve bu algoritmayı **Değişken Komşuluk Arama (Variable Neighborhood Search - VNS)** meta-sezgiseli ile geliştirmeyi amaçlar.

Temel hedef, sınıflandırma doğruluğunu yüksek tutarken, veri setini ayırmak için gereken konik fonksiyon (merkez) sayısını minimize etmektir. Proje, orijinal makaledeki yöntemleri Python ve Gurobi Optimizasyon Çözücüsü kullanarak yeniden üretir ve geliştirir.

## Özellikler

* **r-PCF Algoritması**: İteratif "cookie cutter" (kurabiye kalıbı) mantığıyla çalışan, matematiksel programlama tabanlı sınıflandırma.
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

* **Python 3.10+**
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
| | **VNS-RPCF** | **1.0000** | 0.37 |
| **Breast Cancer** (Sklearn) | r-PCF | 0.9415 | 0.08 |
| | **VNS-RPCF** | **0.9474** | 1.19 |
| **Blobs 3D** | r-PCF | 1.0000 | 0.01 |
| | **VNS-RPCF** | **1.0000** | 0.11 |
| **WBCD** (Diagnosis) | r-PCF | 0.9766 | 0.06 |
| | **VNS-RPCF** | **0.9825** | 0.96 |
| **WBCP** (Prognostic)| r-PCF | **0.7667** | 0.01 |
| | **VNS-RPCF** | 0.7500 | 0.28 |
| **Heart** | r-PCF | 0.6923 | 0.27 |
| | **VNS-RPCF** | **0.7253** | 3.87 |
| **Votes** | r-PCF | **0.9466** | 0.04 |
| | **VNS-RPCF** | **0.9466** | 0.67 |
| **Ionosphere** | r-PCF | 0.9151 | 0.15 |
| | **VNS-RPCF** | **0.9245** | 1.69 |
| **Liver** | r-PCF | **0.9327** | 0.13 |
| | **VNS-RPCF** | 0.9231 | 1.46 |

### Yorumlar

1. **Doğruluk Artışı**: `Heart`, `WBCD`, `Ionosphere` gibi gürültülü veya karmaşık karar sınırlarına sahip veri setlerinde **VNS-RPCF**, standart yönteme göre daha yüksek doğruluk oranlarına ulaşmıştır. Özellikle `Heart` veri setinde yaklaşık **%3.3**'lük bir başarı artışı (0.69 -> 0.72) gözlemlenmiştir. Bu durum, VNS'in yerel arama stratejisinin daha "derin" ve kapsayıcı konik fonksiyonlar bulduğunu doğrular.
2. **Süre Maliyeti**: VNS-RPCF, her iterasyonda optimum merkezi bulmak için komşuluk araması yaptığından eğitim süresi standart r-PCF'e göre belirgin şekilde uzundur. Bu beklenen bir trade-off (ödünleşim) durumudur.
3. **Model Sadeliği**: Her iki algoritma da oldukça az sayıda "merkez" (konik fonksiyon) kullanarak veriyi sınıflandırmayı başarmıştır. Örneğin 3 boyutta %100 ayrım için tek bir koni yeterli olmuştur. Bu, modelin "Sparse" (seyrek) ve yorumlanabilir olduğunu gösterir.

### Sonuç

VNS entegrasyonu, hesaplama maliyetini artırmakla birlikte, özellikle zorlu sınıflandırma problemlerinde modelin genelleme yeteneğini ve doğruluğunu artırmak için etkili bir yöntemdir. Basit veri setlerinde ise standart r-PCF hız avantajı nedeniyle tercih edilebilir.

## Yazar & Referans

Bu uygulama, *kademeli yapıcı (incremental constructive)* öğrenme algoritmaları ve *matematiksel programlama* prensipleri kullanılarak hazırlanmıştır.
