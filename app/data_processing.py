from sklearn.datasets import make_regression

def load_sample_data(n_samples=100, n_features=4, noise=0.1, random_state=42):
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=random_state
    )
    return X, y
