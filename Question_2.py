import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/edyoda/data-science-complete-tutorial/master/Data/house_rental_data.csv.txt"
df = pd.read_csv(url)

columns_drop = ['Index', 'Per_Sqft', 'Location']
existing_columns = set(df.columns)
columns_drop = [i for i in columns_drop if i in existing_columns]


df_cleaned = df.drop(columns=columns_drop, errors='ignore')

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_cleaned)

inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)


plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method to Find Optimal k')
plt.show()

optimal_k = 3 
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(df_scaled)


df.to_csv('clustered_houses.csv', index=False)
