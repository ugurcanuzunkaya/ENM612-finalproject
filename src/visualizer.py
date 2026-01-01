import matplotlib.pyplot as plt
import numpy as np


def plot_decision_boundary(model, X, y, title="r-PCF Decision Boundary"):
    """
    Plots the decision boundary of the r-PCF model along with the dataset.
    Only works for 2D data.
    """
    # Create meshgrid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    resolution = 0.1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, resolution), np.arange(y_min, y_max, resolution)
    )

    # Predict for each point in meshgrid
    # We flatten the meshgrid, predict, and then reshape
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot contours
    plt.figure(figsize=(10, 6))
    # Fill separation: -1 (A) is typically one color, 1 (B) another.
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdBu)

    # Plot data points
    # y is -1 or 1
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor="k", cmap=plt.cm.RdBu)
    plt.colorbar(scatter)

    # Plot Centers
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
    # plt.show() # Don't block execution, maybe save?
