import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

data = {
    "Area": [500, 700, 900, 1100, 1300, 1500, 1700, 1900, 2100, 2300],
    "Price": [25, 32, 40, 48, 55, 63, 70, 78, 85, 92]
}

df = pd.DataFrame(data)

X = df[["Area"]]
y = df["Price"]

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

model = SVR(kernel="rbf", C=100, epsilon=0.1)

model.fit(X_scaled, y_scaled)

y_pred_scaled = model.predict(X_scaled)

y_pred = scaler_y.inverse_transform(
    y_pred_scaled.reshape(-1, 1)
).ravel()

print("Actual Prices:")
print(y.values)

print("\nPredicted Prices:")
print(y_pred)

print("\nMean Squared Error:")
print(mean_squared_error(y, y_pred))

print("\nR2 Score:")
print(r2_score(y, y_pred))

new_area = [[2000]]

new_area_scaled = scaler_X.transform(new_area)

predicted_price_scaled = model.predict(new_area_scaled)

predicted_price = scaler_y.inverse_transform(
    predicted_price_scaled.reshape(-1, 1)
)

print("\nPredicted price for 2000 sq.ft.:")
print(predicted_price[0][0], "Lakh")

plt.scatter(X, y, label="Actual Data")
plt.plot(X, y_pred, label="SVR Prediction")

plt.xlabel("House Area (sq.ft.)")
plt.ylabel("House Price (Lakh)")
plt.title("Support Vector Regression")

plt.legend()
plt.grid(True)
plt.show()