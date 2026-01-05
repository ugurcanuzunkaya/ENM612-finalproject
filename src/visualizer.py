import matplotlib.pyplot as plt
import numpy as np


def plot_decision_boundary(model, X, y, title="r-PCF Decision Boundary"):
    """
    r-PCF modelinin karar sınırını veri seti ile birlikte çizer.
    Sadece 2D veriler için çalışır.
    """
    # Ağ ızgarası (meshgrid) oluştur
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    resolution = 0.1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, resolution), np.arange(y_min, y_max, resolution)
    )

    # Meshgrid'deki her nokta için tahmin yap
    # Meshgrid'i düzleştiriyor, tahmin yapıyor ve yeniden şekillendiriyoruz
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Konturları çiz
    plt.figure(figsize=(10, 6))
    # Ayırmayı doldur: -1 (A) tipik olarak bir renktir, 1 (B) başka bir renktir.
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdBu)

    # Veri noktalarını çiz
    # y, -1 veya 1'dir
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor="k", cmap=plt.cm.RdBu)
    plt.colorbar(scatter)

    # Merkezleri Çiz
    if hasattr(model, "centers") and len(model.centers) > 0:
        centers = np.array(model.centers)
        plt.scatter(
            centers[:, 0],
            centers[:, 1],
            c="yellow",
            s=150,
            marker="*",
            edgecolors="black",
            label="Centers",
        )

    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    # plt.show() # Yürütmeyi engelleme, belki kaydet?
