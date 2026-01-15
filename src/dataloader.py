"""
Data Loading Module.

This module is responsible for fetching, loading, and preprocessing various datasets
from sklearn and the UCI Machine Learning Repository. It handles normalization
(StandardScaler) and label encoding for compatibility with r-PCF methodology.
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
    Central handler for loading and preprocessing datasets.
    Supports both synthetic (sklearn) and real-world (UCI) datasets.
    """

    def __init__(self):
        self.scaler = StandardScaler()

    def load_dataset(self, dataset_name):
        """
        Routes to the specific dataset loader by name.
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
        elif dataset_name == "statlog_heart":
            return self.load_statlog_heart()
        elif dataset_name == "abalone":
            return self.load_abalone()
        elif dataset_name == "spambase":
            return self.load_spambase()
        else:
            raise ValueError(
                f"Dataset '{dataset_name}' not found. Available: moons, breast_cancer, blobs_3d, wbcd, wbcp, heart, liver, votes, ionosphere, statlog_heart, abalone, spambase"
            )

    def load_spambase(self):
        """
        Loads the Spambase Data Set.
        UCI Repo ID: 94
        Target: Spam (1) or Not Spam (0).
        """
        print("\\n--- Loading 'Spambase' Dataset [via ucimlrepo] ---")
        if fetch_ucirepo is None:
            raise ImportError("ucimlrepo not installed")

        try:
            spambase = fetch_ucirepo(id=94)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch Spambase dataset: {e}")

        X_df = spambase.data.features
        y_df = spambase.data.targets

        # Convert to numpy
        X = X_df.values
        y = y_df.values.ravel()

        # Handle Missing Values (Spambase usually clean but good practice)
        imputer = SimpleImputer(strategy="mean")
        X = imputer.fit_transform(X)

        # Scale features
        if hasattr(self, "scaler") and self.scaler:
            X = self.scaler.fit_transform(X)

        return X, y

    def load_abalone(self):
        """
        Loads the Abalone Data Set.
        UCI Repo ID: 1
        Target: Rings (Integer).
        Preprocessing:
          - 'Sex' feature (M, F, I) is encoded.
          - 'Rings' is binarized to create a Classification task (<=9 vs >9).
        """
        print("\n--- Loading 'Abalone' Dataset [via ucimlrepo] ---")
        if fetch_ucirepo is None:
            raise ImportError("ucimlrepo not installed")

        try:
            abalone = fetch_ucirepo(id=1)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch Abalone dataset: {e}")

        X_df = abalone.data.features
        y_df = abalone.data.targets

        # 1. Handle Categorical Feature 'Sex'
        # Check if 'Sex' is in columns (it should be)
        if "Sex" in X_df.columns:
            # Simple Label Encoding or One-Hot. Let's use LabelEncoder for simplicity
            # as it keeps dimensionality lower, although One-Hot is theoretically better.
            # Given r-PCF works with distances, One-Hot might be safer?
            # Let's use pd.get_dummies if possible, but we are using numpy outputs.
            # Let's just map M, F, I manually or use LabelEncoder.
            le_sex = LabelEncoder()
            X_df = X_df.copy()
            X_df["Sex"] = le_sex.fit_transform(X_df["Sex"])

        # Convert to numpy
        X = X_df.values
        y = y_df.values.ravel()

        # 2. Binarize Target 'Rings'
        # Median is typically around 9. We will split: Class 0 <= 9, Class 1 > 9
        # This converts it into a binary classification problem "Young vs Old"
        threshold = 9
        y = np.where(y > threshold, 1, 0)

        # 3. Scaling
        if hasattr(self, "scaler") and self.scaler:
            X = self.scaler.fit_transform(X)

        return X, y

    def load_statlog_heart(self):
        """
        Loads the Statlog (Heart) Data Set.
        UCI Repo ID: 145
        Target: Presence of heart disease
        """
        print("\n--- Loading 'Statlog Heart' Dataset [via ucimlrepo] ---")
        if fetch_ucirepo is None:
            raise ImportError("ucimlrepo not installed")

        # fetch dataset
        try:
            statlog_heart = fetch_ucirepo(id=145)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch Statlog Heart dataset: {e}")

        # data (as pandas dataframes)
        X = statlog_heart.data.features
        y = statlog_heart.data.targets

        # Convert to numpy
        X = X.values
        y = y.values.ravel()

        # Handle Missing Values (if any)
        imputer = SimpleImputer(strategy="mean")
        X = imputer.fit_transform(X)

        # Encode Targets if necessary (Statlog heart is typically 1 (absence) / 2 (presence))
        # We want binary 0/1 or -1/+1
        le = LabelEncoder()
        y = le.fit_transform(y)  # Maps to 0/1

        # Scaling
        if hasattr(self, "scaler") and self.scaler:
            X = self.scaler.fit_transform(X)

        return X, y

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

        # Encode target (M/B -> 1/0)
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

        # Drop 'Time' column if only features are needed (optional, depends on target)
        # Typically 'Time' is excluded for pure classification of recurrence
        # X = X[:, 1:]

        # Encode target (R/N -> 1/0)
        le = LabelEncoder()
        y = le.fit_transform(y)

        # Fill NaN values if present (WBCP has missing values in 'Lymph node status' etc.)
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

        # Fill missing values (may be NaN in new repo)
        imputer = SimpleImputer(strategy="mean")
        X = imputer.fit_transform(X)

        # Binarize target: 0 is healthy, >0 is heart disease
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

        # Target in BUPA is typically 'selector' (field 7).
        # Ensure encoded as 0/1 (may initially be 1/2).
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

        # Handle '?' or NaN in features (Voting records have many abstentions)
        # ucimlrepo typically loads these as NaN.
        # Strategy: Fill with 'most_frequent' or treat as separate category.

        # Map 'y'/'n' values to 1/0 manually if strings
        if hasattr(X, "replace"):
            X = X.replace({"y": 1, "n": 0, "?": np.nan})

        imputer = SimpleImputer(strategy="most_frequent")
        X = imputer.fit_transform(X)

        y = y.values.ravel()

        # Encode target (democrat/republican -> 0/1)
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

        # Encode target
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
        1. Load your data (e.g., CSV, Excel, or from a library).
        2. Separate features (X) and target (y).
        3. Ensure X is a numpy array of shape (n_samples, n_features).
        4. Ensure y is a numpy array of shape (n_samples,).
        5. Apply scaling if needed.
        6. Return X, y.
        """
        print("\n--- Loading 'Custom' Dataset ---")
        # --- YOUR CODE HERE ---
        # Example:
        # data = pd.read_csv("my_data.csv")
        # X = data.drop("target", axis=1).values
        # y = data["target"].values

        # Placeholder (Replace with your data)
        X = np.random.randn(100, 5)  # 100 samples, 5 features
        y = np.random.randint(0, 2, 100)  # Binary target

        # Scaling
        X = self.scaler.fit_transform(X)

        return X, y
