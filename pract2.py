import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = {
    'Income': [20, 22, 25, 30, 35, 60, 65, 70, 75, 80],
    'Spending': [80, 75, 78, 70, 65, 40, 35, 30, 25, 20]
}

df = pd.DataFrame(data)

X = df[['Income', 'Spending']]

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

kmeans.fit(X)

df['Cluster'] = kmeans.labels_

print(df)

print("\nCluster Centers:")
print(kmeans.cluster_centers_)

# Plot the clusters
plt.scatter(
    df['Income'],
    df['Spending'],
    c=df['Cluster']
)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("K-Means Clustering")
plt.show()