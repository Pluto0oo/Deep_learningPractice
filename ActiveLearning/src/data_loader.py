import os
import re
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def load_data(config: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    data_config = config["data"]
    data_path = data_config.get("path", "data/processed")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path not found: {data_path}")
    
    X = np.load(os.path.join(data_path, "X.npy"))
    y = np.load(os.path.join(data_path, "y.npy"))
    
    test_size = data_config.get("test_size", 0.2)
    random_state = config["experiment"].get("seed", 42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    val_size = data_config.get("val_size", 0.1)
    if val_size > 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=random_state, stratify=y_train
        )
    else:
        X_val, y_val = None, None
    
    train_data = {"X": X_train, "y": y_train}
    test_data = {"X": X_test, "y": y_test}
    
    if X_val is not None:
        train_data["X_val"] = X_val
        train_data["y_val"] = y_val
    
    return train_data, test_data


def create_sample_data(data_path: str, n_samples: int = 1000, n_features: int = 20, n_classes: int = 5) -> None:
    os.makedirs(data_path, exist_ok=True)
    
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)
    
    np.save(os.path.join(data_path, "X.npy"), X)
    np.save(os.path.join(data_path, "y.npy"), y)
    print(f"Sample data created at {data_path}")


def get_data_stats(data: Dict[str, np.ndarray]) -> Dict[str, Any]:
    stats = {
        "n_samples": data["X"].shape[0],
        "n_features": data["X"].shape[1],
        "n_classes": len(np.unique(data["y"])),
        "class_distribution": dict(zip(*np.unique(data["y"], return_counts=True))),
    }
    
    if "X_val" in data:
        stats["n_val_samples"] = data["X_val"].shape[0]
    
    return stats


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_imdb_data(data_path: str = "data/raw/imdb", max_features: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
    import tarfile
    import requests
    from tqdm import tqdm
    
    imdb_url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    tar_path = os.path.join(data_path, "aclImdb_v1.tar.gz")
    
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
    
    if not os.path.exists(tar_path):
        print("Downloading IMDB dataset...")
        try:
            response = requests.get(imdb_url, stream=True, timeout=300)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            
            with open(tar_path, 'wb') as f, tqdm(
                desc='Downloading IMDB',
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    bar.update(size)
        except Exception as e:
            print(f"Failed to download IMDB dataset: {e}")
            print("Using simulated data instead...")
            return generate_simulated_data(n_samples=5000, n_features=max_features, seed=42)
    
    extracted_dir = os.path.join(data_path, "aclImdb")
    if not os.path.exists(extracted_dir):
        print("Extracting IMDB dataset...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(data_path)
    
    texts = []
    labels = []
    
    for label, folder in [(0, "neg"), (1, "pos")]:
        folder_path = os.path.join(extracted_dir, "train", folder)
        for filename in os.listdir(folder_path):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                texts.append(preprocess_text(f.read()))
            labels.append(label)
        
        folder_path = os.path.join(extracted_dir, "test", folder)
        for filename in os.listdir(folder_path):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                texts.append(preprocess_text(f.read()))
            labels.append(label)
    
    print(f"Loaded {len(texts)} IMDB reviews")
    
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    X = vectorizer.fit_transform(texts).toarray()
    y = np.array(labels)
    
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    np.save(os.path.join(processed_dir, "X_imdb.npy"), X)
    np.save(os.path.join(processed_dir, "y_imdb.npy"), y)
    print(f"IMDB data saved to {processed_dir}")
    
    return X, y


def load_imdb_for_active_learning(
    seed: int = 42,
    test_size: float = 0.2,
    initial_label_ratio: float = 0.05,
    max_features: int = 5000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    processed_dir = "data/processed"
    x_path = os.path.join(processed_dir, "X_imdb.npy")
    y_path = os.path.join(processed_dir, "y_imdb.npy")
    
    if os.path.exists(x_path) and os.path.exists(y_path):
        X = np.load(x_path)
        y = np.load(y_path)
        print("Loaded IMDB data from processed directory")
    else:
        X, y = load_imdb_data(max_features=max_features)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    
    n_initial = int(len(X_train) * initial_label_ratio)
    indices = np.random.RandomState(seed).permutation(len(X_train))
    
    X_initial = X_train[indices[:n_initial]]
    y_initial = y_train[indices[:n_initial]]
    
    X_pool = X_train[indices[n_initial:]]
    y_pool = y_train[indices[n_initial:]]
    
    print(f"Initial labeled: {len(X_initial)}, Pool size: {len(X_pool)}, Test size: {len(X_test)}")
    
    return X_initial, y_initial, X_pool, y_pool, X_test, y_test


def generate_simulated_data(n_samples: int = 5000, n_features: int = 500, seed: int = 42):
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] > 0).astype(int)
    
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    np.save(os.path.join(processed_dir, "X_imdb.npy"), X)
    np.save(os.path.join(processed_dir, "y_imdb.npy"), y)
    
    return X, y