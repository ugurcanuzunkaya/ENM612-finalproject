import matplotlib.pyplot as plt
import numpy as np


def plot_decision_boundary(model, X, y, title="r-PCF Decision Boundary"):
    """
    Plots the r-PCF decision boundary, data points (Set A/B), and individual conic functions.
    Works only for 2D data.
    """
    # Create meshgrid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    resolution = 0.05
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, resolution), np.arange(y_min, y_max, resolution)
    )

    # Predict for each point in meshgrid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 6))

    # Plot Overall Decision Region
    # -1 (Set A) vs 1 (Set B)
    # We use a light colormap
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

    # Plot Overall Decision Boundary (Black Line)
    plt.contour(xx, yy, Z, levels=[0], colors="black", linewidths=2, linestyles="solid")

    # Plot Individual Conic Functions (g_k(x) = 0)
    # Use a colormap for distinct colors
    colormap = plt.cm.get_cmap("tab10")

    if hasattr(model, "functions"):
        for i, func in enumerate(model.functions):
            # Calculate g(x) for the grid
            pts = np.c_[xx.ravel(), yy.ravel()]
            diff = pts - func["center"]
            term1 = np.dot(diff, func["w"])
            term2 = func["xi"] * np.sum(np.abs(diff), axis=1)
            g_vals = term1 + term2 - func["gamma"]
            Z_func = g_vals.reshape(xx.shape)

            # Plot the zero contour with a unique color
            color = colormap(i % 10)
            plt.contour(
                xx,
                yy,
                Z_func,
                levels=[0],
                colors=[color],
                linestyles="dashed",
                linewidths=1.5,
                alpha=0.9,
            )

        # Add proxy artists for the legend
        plt.plot(
            [],
            [],
            color="black",
            linewidth=2,
            linestyle="solid",
            label="Global Boundary",
        )
        plt.plot(
            [],
            [],
            color="gray",
            linestyle="dashed",
            linewidth=1.5,
            label="Conic Functions (Multiple Colors)",
        )

    # Plot Data Points (Set A vs Set B)
    # y = -1 -> Set A (Blue-ish), y = 1 -> Set B (Red-ish)

    # Set A (-1)
    idx_A = np.where(y == -1)[0]
    plt.scatter(
        X[idx_A, 0],
        X[idx_A, 1],
        c="blue",
        s=30,
        edgecolor="k",
        label="Set A (Class -1)",
        marker="o",
        alpha=0.8,
    )

    # Set B (1)
    idx_B = np.where(y == 1)[0]
    plt.scatter(
        X[idx_B, 0],
        X[idx_B, 1],
        c="red",
        s=30,
        edgecolor="k",
        label="Set B (Class +1)",
        marker="s",
        alpha=0.8,
    )

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
            zorder=10,
        )

    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend(loc="best")
    # plt.show()


def plot_decision_boundary_3d(model, X, y, title="r-PCF 3D Visualization"):
    """
    Plots the r-PCF decision boundary and data points in 3D.
    Note: 'Show' mode requires an interactive backend.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot Data Points
    # Set A (-1)
    idx_A = np.where(y == -1)[0]
    ax.scatter(
        X[idx_A, 0],
        X[idx_A, 1],
        X[idx_A, 2],
        c="blue",
        marker="o",
        label="Set A (-1)",
        alpha=0.6,
    )

    # Set B (1)
    idx_B = np.where(y == 1)[0]
    ax.scatter(
        X[idx_B, 0],
        X[idx_B, 1],
        X[idx_B, 2],
        c="red",
        marker="^",
        label="Set B (+1)",
        alpha=0.6,
    )

    # Plot Centers
    if hasattr(model, "centers") and len(model.centers) > 0:
        centers = np.array(model.centers)
        ax.scatter(
            centers[:, 0],
            centers[:, 1],
            centers[:, 2],
            c="yellow",
            s=100,
            marker="*",
            edgecolors="black",
            label="Centers",
        )

    # Visualizing the boundary in 3D is complex without marching cubes.
    # We will approximate it by scattering points where the decision function is close to 0.

    # Create a sparse grid
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    z_min, z_max = X[:, 2].min(), X[:, 2].max()

    # Grid resolution (adjust for performance vs quality)
    res = 15j
    xx, yy, zz = np.mgrid[x_min:x_max:res, y_min:y_max:res, z_min:z_max:res]

    grid_pts = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

    # Predict/Evaluate g(x)
    # We want g_min(x) approx 0
    if hasattr(model, "functions"):
        g_matrix = np.zeros((len(grid_pts), len(model.functions)))
        for k, func in enumerate(model.functions):
            diff = grid_pts - func["center"]
            term1 = np.dot(diff, func["w"])
            term2 = func["xi"] * np.sum(np.abs(diff), axis=1)
            g_matrix[:, k] = term1 + term2 - func["gamma"]

        g_min = np.min(g_matrix, axis=1)

        # Select points close to 0
        epsilon = 0.5
        boundary_mask = np.abs(g_min) < epsilon

        if np.any(boundary_mask):
            xb = grid_pts[boundary_mask, 0]
            yb = grid_pts[boundary_mask, 1]
            zb = grid_pts[boundary_mask, 2]

            ax.scatter(xb, yb, zb, c="green", alpha=0.3, s=5, label="Approx Boundary")

    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_zlabel("Feature 3")
    ax.legend()

    plt.show()  # Show directly as requested
