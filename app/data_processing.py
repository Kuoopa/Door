from sklearn.datasets import make_classification

def load_sample_data(n_samples=100, n_features=4, n_classes=2, random_state=42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        random_state=random_state
    )
    return X, y
