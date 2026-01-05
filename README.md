# Revised Polyhedral Conic Functions (r-PCF) ve VNS İyileştirmesi

## Proje Hedefi

Bu proje, ENM612 final projesi kapsamında, ikili sınıflandırma (binary classification) problemleri için geliştirilen **r-PCF (Revised Polyhedral Conic Functions)** algoritmasını uygulamayı ve bu algoritmayı **Değişken Komşuluk Arama (Variable Neighborhood Search - VNS)** meta-sezgiseli ile geliştirmeyi amaçlar.

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

Aşağıdaki tablo, farklı veri setleri üzerinde standart **r-PCF** ve **VNS-RPCF** algoritmalarının performans karşılaştırmasını göstermektedir. (Sonuçlar `np.random.seed(42)` ile alınmıştır).

| Veri Seti | Model | Doğruluk (Accuracy) | Süre (sn) | Fonksiyon Sayısı |
| :--- | :--- | :--- | :--- | :--- |
| **Moons** | r-PCF | 0.9833 | 0.0156 | 5 |
| | **VNS-RPCF** | **1.0000** | 0.3206 | 6 |
| **Breast Cancer** (Sklearn) | r-PCF | **0.9649** | 0.0971 | 2 |
| | **VNS-RPCF** | 0.9357 | 1.5493 | 3 |
| **Blobs 3D** | r-PCF | 1.0000 | 0.0064 | 1 |
| | **VNS-RPCF** | 1.0000 | 0.1123 | 1 |
| **WBCD** (Diagnosis) | r-PCF | 0.9357 | 0.123 | 4 |
| | **VNS-RPCF** | **0.9474** | 1.4736 | 3 |
| **WBCP** (Prognostic) | r-PCF | 0.7833 | 0.015 | 3 |
| | **VNS-RPCF** | 0.7833 | 0.2705 | 3 |
| **Heart** | r-PCF | 0.7692 | 0.0302 | 3 |
| | **VNS-RPCF** | **0.7802** | 0.3684 | 3 |
| **Votes** | r-PCF | 0.9618 | 0.0875 | 11 |
| | **VNS-RPCF** | 0.9618 | 1.0907 | **5** |
| **Ionosphere** | r-PCF | 0.9151 | 0.1299 | 5 |
| | **VNS-RPCF** | **0.9528** | 1.5615 | 5 |
| **Liver** | r-PCF | 0.9038 | 0.1042 | 4 |
| | **VNS-RPCF** | **0.9615** | 1.5208 | 4 |

### Yorumlar

1. **Doğruluk Artışı**: `Liver`, `Ionosphere`, `Moons`, `WBCD` ve `Heart` veri setlerinde **VNS-RPCF**, standart yönteme göre daha yüksek doğruluk sağlamıştır. Özellikle `Liver` veri setinde yaklaşık **%6**'lık (0.90 -> 0.96) ve `Ionosphere` setinde **%4**'lük belirgin bir artış görülmüştür.
2. **Model Sadeliği**: `Votes` veri setinde her iki algoritma aynı doğruluğu (0.9618) yakalamış olsa da, **VNS-RPCF** bu sonuca sadece **5** konik fonksiyon ile ulaşırken, standart r-PCF **11** fonksiyon kullanmıştır. Bu durum, VNS'in daha sade (sparse) ve daha iyi genelleme yapan modeller bulabildiğini gösterir. Aynı şekilde `WBCD` setinde de daha az merkez ile daha yüksek doğruluk elde edilmiştir.
3. **Süre Maliyeti**: VNS-RPCF, her iterasyonda optimum merkezi bulmak için komşuluk araması yaptığından eğitim süresi standart r-PCF'e göre daha uzundur. Ancak karmaşık problemler ve daha sade modeller için bu maliyet kabul edilebilir düzeydedir.
4. **Tutarlılık**: `Blobs 3D` ve `WBCP` gibi setlerde her iki algoritma da benzer sonuçlar üretmiştir, bu da VNS'in temel algoritmanın kararlılığını bozmadığını gösterir.

### Sonuç

VNS entegrasyonu, hesaplama maliyetini bir miktar artırmakla birlikte, modelin sınıflandırma doğruluğunu artırma (Liver, Ionosphere) ve model karmaşıklığını azaltma (Votes) konularında standart r-PCF'e göre üstünlük sağlamaktadır. Özellikle zorlu ve gürültülü veri setlerinde VNS-RPCF tercih edilmelidir.
