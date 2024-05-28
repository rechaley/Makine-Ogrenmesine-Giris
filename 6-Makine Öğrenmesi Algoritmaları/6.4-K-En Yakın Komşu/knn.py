import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Örnek veri seti oluşturma
X, y = make_blobs(n_samples=100, centers=3, random_state=42)

# Veri setini eğitim ve test kümelerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN modelini oluşturma ve eğitme
k = 3  # K değeri (komşu sayısı)
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Test veri seti üzerinde tahmin yapma
y_pred = knn.predict(X_test)

# Doğruluk skorunu hesaplama
accuracy = accuracy_score(y_test, y_pred)
print(f"Doğruluk: {accuracy:.2f}")

# Sınıfları ve karar sınırlarını görselleştirme
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolors='k')
plt.title(f'K-En Yakın Komşu (K={k})')
plt.xlabel('Özellik 1')
plt.ylabel('Özellik 2')
plt.show()
