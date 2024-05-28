import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Basit veri kümesi oluşturma
data = {
    'mileage': [50000, 30000, 40000, 80000, 60000],
    'age': [5, 3, 4, 8, 6],
    'price': [15000, 20000, 18000, 12000, 14000]
}
df = pd.DataFrame(data)

# Özellikler (X) ve hedef değişken (y)
X = df[['mileage', 'age']]
y = df['price']

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model oluşturma ve eğitme
model = LinearRegression()
model.fit(X_train, y_train)

# Tahmin yapma
y_pred = model.predict(X_test)

# Model performansını değerlendirme
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Yeni bir veri noktası için tahmin yapma
new_car = {'mileage': 35000, 'age': 4}
new_car_df = pd.DataFrame([new_car])
new_price = model.predict(new_car_df)
print(f'Tahmin edilen fiyat: ${new_price[0]:,.2f}')