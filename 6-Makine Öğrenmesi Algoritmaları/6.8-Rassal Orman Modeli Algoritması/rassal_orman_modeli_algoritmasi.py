from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Verisetini yükle
data = load_iris()
X = data.data
y = data.target

# Eğitim ve test verilerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Rassal Orman modelini oluştur
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Modeli eğit
model.fit(X_train, y_train)

# Tahminlerde bulun
y_pred = model.predict(X_test)

# Modelin doğruluğunu hesapla
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
