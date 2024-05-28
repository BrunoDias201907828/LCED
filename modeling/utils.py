import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans


class KMeansTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, features_indexes: tuple[int, ...]):
        self.features_indexes = features_indexes
        self.kmeans = KMeans(n_clusters=2)

    def fit(self, X, y=None):
        self.kmeans.fit(X[:, self.features_indexes])
        return self

    def transform(self, X):
        cluster_labels = self.kmeans.predict(X[:, self.features_indexes])
        X_transformed = np.concatenate([X, cluster_labels.reshape(-1, 1)], axis=1)
        return X_transformed