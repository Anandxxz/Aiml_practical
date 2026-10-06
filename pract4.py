import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

iris = load_iris()
X = iris.data

y = iris.target

print("Feature Names:")
print(iris.feature_names)

print("\nClass Names:")
print(iris.target_names)

df = pd.DataFrame(
    X,
    columns=iris.feature_names
)
df["Target"] = y

print("\nFirst 5 rows of the dataset:")
print(df.head())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

model = GaussianNB()
model.fit(X_train, y_train)

print("\nModel training completed.")

y_pred = model.predict(X_test)

print("\nPredicted Classes:")
print(y_pred)

print("\nActual Classes:")
print(y_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(accuracy)
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names
    )
)

new_flower = []
prediction = model.predict(new_flower)
print("\nPrediction for New Flower:")
print(iris.target_names[prediction[0]])