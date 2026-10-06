import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

wine = load_wine()

X = wine.data
y = wine.target

feature_names = wine.feature_names

df = pd.DataFrame(X, columns=feature_names)

print("First 5 rows of the dataset:")
print(df.head())

print("\nShape of original dataset:")
print(X.shape)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nData after standardization:")
print(X_scaled[:5])

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(
    X_pca,
    columns=['Principal Component 1',
             'Principal Component 2']
)

pca_df['Class'] = y

print("\nPCA Result:")
print(pca_df.head())

explained_variance = pca.explained_variance_ratio_

print("\nExplained Variance Ratio:")
print(explained_variance)

print("\nTotal Variance Retained:")
print(explained_variance.sum())

plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=y
)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA - Wine Dataset")
plt.colorbar(label="Wine Class")
plt.grid(True)
plt.show()