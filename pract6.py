import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = {
    "Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "Marks": [35, 40, 50, 55, 65, 70, 78, 85]
}

df = pd.DataFrame(data)

X = df[["Study_Hours"]]
y = df["Marks"]

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

print("\nPredicted Marks:")
print(y_pred)

print("\nMean Squared Error:", mean_squared_error(y, y_pred))
print("R2 Score:", r2_score(y, y_pred))

study_hours = [[9]]

predicted_marks = model.predict(study_hours)

print("\nPredicted marks for 9 hours of study:")
print(predicted_marks[0])

plt.scatter(X, y)
plt.plot(X, y_pred)
plt.xlabel("Study Hours")
plt.ylabel("Exam Marks")
plt.title("Linear Regression: Study Hours vs Exam Marks")
plt.grid(True)
plt.show()