# Gerekli kütüphanelerin import edilmesi
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Iris veri setinin yüklenmesi
iris = load_iris()
X = iris.data

# Verilerin standartlaştırılması
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-means modeli oluşturulması
kmeans = KMeans(n_clusters=3, random_state=42)

# Modelin eğitilmesi (kümelerin bulunması)
kmeans.fit(X_scaled)

# Küme etiketlerinin tahmin edilmesi
labels = kmeans.labels_

# Küme merkezlerinin elde edilmesi
centers = kmeans.cluster_centers_

# Sonuçların görselleştirilmesi
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k')
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='x')
plt.title('K-means Kümeleme (Iris Veriseti)')
plt.xlabel('Özellik 1')
plt.ylabel('Özellik 2')
plt.show()
