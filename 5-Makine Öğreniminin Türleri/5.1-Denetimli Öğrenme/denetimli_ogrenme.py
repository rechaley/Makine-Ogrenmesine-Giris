# Gerekli kütüphanelerin import edilmesi
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Iris veri setinin yüklenmesi
iris = load_iris()
X = iris.data  # Özellikler (features)
y = iris.target  # Etiketler (labels)

# Eğitim ve test setlerine ayırma (70% eğitim, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Karar ağacı sınıflandırıcısının oluşturulması
clf = DecisionTreeClassifier()

# Modelin eğitim verisi ile eğitilmesi
clf.fit(X_train, y_train)

# Test verisi ile tahmin yapma
y_pred = clf.predict(X_test)

# Modelin doğruluk (accuracy) skorunun hesaplanması
accuracy = accuracy_score(y_test, y_pred)

# Sonuçların yazdırılması
print(f"Model Doğruluğu: {accuracy * 100:.2f}%")
