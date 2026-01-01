import numpy as np
import pandas as pd
from sklearn.datasets import make_moons, load_breast_cancer, make_blobs
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

try:
    from ucimlrepo import fetch_ucirepo
except ImportError:
    fetch_ucirepo = None


class DatasetLoader:
    """
    Handles loading and preprocessing of datasets.
    """

    def __init__(self):
        self.scaler = StandardScaler()

    def load_dataset(self, dataset_name):
        """
        Dispatches to the specific dataset loader based on the name.
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
        elif dataset_name == "custom":
            return self.load_custom_dataset()
        else:
            raise ValueError(
                f"Dataset '{dataset_name}' not found. Available: moons, breast_cancer, blobs_3d, wbcd, wbcp, heart, liver, votes, ionosphere, custom"
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
        Loads the Wisconsin Breast Cancer Diagnosis (WBCD) dataset.
        UCI Repo ID: 17
        Target: Diagnosis (M = malignant, B = benign)
        """
        print("\n--- Loading 'WBCD' (Diagnosis) Dataset [via ucimlrepo] ---")
        if fetch_ucirepo is None:
            raise ImportError("ucimlrepo not installed")

        # Fetch dataset
        dataset = fetch_ucirepo(id=17)

        # Extract features and targets
        X = dataset.data.features
        y = dataset.data.targets

        # Convert to numpy arrays
        X = X.values
        y = y.values.ravel()  # Flatten to shape (n_samples,)

        # Encode Target (M/B -> 1/0)
        le = LabelEncoder()
        y = le.fit_transform(y)

        # Scaling
        if hasattr(self, "scaler") and self.scaler:
            X = self.scaler.fit_transform(X)

        return X, y

    def load_wbcp(self):
        """
        Loads the Wisconsin Breast Cancer Prognosis (WBCP) dataset.
        UCI Repo ID: 16
        Target: Outcome (R = recur, N = nonrecur)
        """
        print("\n--- Loading 'WBCP' (Prognosis) Dataset [via ucimlrepo] ---")
        if fetch_ucirepo is None:
            raise ImportError("ucimlrepo not installed")

        # Fetch dataset
        dataset = fetch_ucirepo(id=16)

        X = dataset.data.features
        y = dataset.data.targets

        # Preprocessing
        X = X.values
        y = y.values.ravel()

        # Drop the 'Time' column if you only want features (optional, depending on goal)
        # Often 'Time' is excluded for pure classification of recurrence
        # X = X[:, 1:]

        # Encode Target (R/N -> 1/0)
        le = LabelEncoder()
        y = le.fit_transform(y)

        # Impute NaNs if any (WBCP has missing values in 'Lymph node status' etc)
        imputer = SimpleImputer(strategy="mean")
        X = imputer.fit_transform(X)

        # Scaling
        if hasattr(self, "scaler") and self.scaler:
            X = self.scaler.fit_transform(X)

        return X, y

    def load_heart(self):
        """
        Loads the Cleveland Heart Disease dataset.
        UCI Repo ID: 45
        Target: Diagnosis (0=healthy, 1-4=sick)
        """
        print("\n--- Loading 'Cleveland Heart' Dataset [via ucimlrepo] ---")
        if fetch_ucirepo is None:
            raise ImportError("ucimlrepo not installed")

        # Fetch dataset (ID 45 is the main Heart Disease container)
        dataset = fetch_ucirepo(id=45)

        X = dataset.data.features
        y = dataset.data.targets

        # Convert to numpy
        X = X.values
        y = y.values.ravel()

        # Impute missing values (The new repo might have NaNs)
        imputer = SimpleImputer(strategy="mean")
        X = imputer.fit_transform(X)

        # Binarize Target: 0 is healthy, >0 is heart disease
        y = np.where(y > 0, 1, 0)

        # Scaling
        if hasattr(self, "scaler") and self.scaler:
            X = self.scaler.fit_transform(X)

        return X, y

    def load_liver(self):
        """
        Loads the BUPA Liver Disorders dataset.
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

        # The target in BUPA is often 'selector' (field 7).
        # Ensure it is encoded to 0/1 (it might be 1/2 originally).
        le = LabelEncoder()
        y = le.fit_transform(y)

        # Scaling
        if hasattr(self, "scaler") and self.scaler:
            X = self.scaler.fit_transform(X)

        return X, y

    def load_votes(self):
        """
        Loads the U.S. Congressional Voting Records dataset.
        UCI Repo ID: 105
        Target: Party (Democrat/Republican)
        """
        print("\n--- Loading 'Congress Voting' Dataset [via ucimlrepo] ---")
        if fetch_ucirepo is None:
            raise ImportError("ucimlrepo not installed")

        dataset = fetch_ucirepo(id=105)

        X = dataset.data.features
        y = dataset.data.targets

        # Handling '?' or NaNs in features (Voting records have many abstentions)
        # ucimlrepo loads them as NaNs usually.
        # Strategy: Impute with 'most_frequent' or treat as a separate category.

        # Manually map 'y'/'n' to 1/0 if they are strings
        if hasattr(X, "replace"):
            X = X.replace({"y": 1, "n": 0, "?": np.nan})

        imputer = SimpleImputer(strategy="most_frequent")
        X = imputer.fit_transform(X)

        y = y.values.ravel()

        # Encode Target (democrat/republican -> 0/1)
        le = LabelEncoder()
        y = le.fit_transform(y)

        # Scaling
        if hasattr(self, "scaler") and self.scaler:
            X = self.scaler.fit_transform(X)

        return X, y

    def load_ionosphere(self):
        """
        Loads the Ionosphere dataset.
        UCI Repo ID: 52
        Target: Class (g=good, b=bad)
        """
        print("\n--- Loading 'Ionosphere' Dataset [via ucimlrepo] ---")
        if fetch_ucirepo is None:
            raise ImportError("ucimlrepo not installed")

        dataset = fetch_ucirepo(id=52)

        X = dataset.data.features
        y = dataset.data.targets

        X = X.values
        y = y.values.ravel()

        # Encode Target
        le = LabelEncoder()
        y = le.fit_transform(y)

        # Scaling
        if hasattr(self, "scaler") and self.scaler:
            X = self.scaler.fit_transform(X)

        return X, y

    def load_custom_dataset(self):
        """
        TEMPLATE: Use this method to add your own dataset.

        Steps:
        1. Load your data (e.g., from CSV, Excel, or library).
        2. Separate into features (X) and target (y).
        3. Ensure X is a numpy array of shape (n_samples, n_features).
        4. Ensure y is a numpy array of shape (n_samples,).
        5. Apply scaling if necessary.
        6. Return X, y.
        """
        print("\n--- Loading 'Custom' Dataset ---")
        # --- YOUR CODE HERE ---
        # Example:
        # data = pd.read_csv("my_data.csv")
        # X = data.drop("target", axis=1).values
        # y = data["target"].values

        # Placeholder (Replace this with your data)
        X = np.random.randn(100, 5)  # 100 samples, 5 features
        y = np.random.randint(0, 2, 100)  # Binary target

        # Scaling
        X = self.scaler.fit_transform(X)

        return X, y
