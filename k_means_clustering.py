# K-Means Clustering
"""
Created on Tue Feb 19 11:31:49 2019

@author: Chad Gouws
"""

import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('Data/K Means Clustering/CC GENERAL.xls', sep=',')           # Read data from CSV
df = pd.DataFrame(data)

missing = data.isna().sum()                                                     # Check what data is missing
print(missing)

data = data.fillna(data.median())                                               # Fill missing data with median

vals = data.iloc[:, 1:].values                                                  # Exclude customer ID

from sklearn.cluster import KMeans

wcss = []                                           # The goal is to minimize the WCSS (within-cluster sum of squares)

for i in range(1, 30):
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=200)        # Perform clustering
    kmeans.fit_predict(vals)
    wcss.append(kmeans.inertia_)
    
plt.plot(wcss, 'ro-', label='WCSS')                                             # Plot WCSS
plt.title('Computing WCSS for KMeans++')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

import seaborn as sbn

best_cols = ['INSTALLMENTS_PURCHASES', 'CREDIT_LIMIT', 'MINIMUM_PAYMENTS', 'ONEOFF_PURCHASES']      # Select best correlations
kmeans = KMeans(n_clusters=8, init='k-means++', n_init=10, max_iter=300)
best_vals = data[best_cols].iloc[:, 1:].values
y_pred = kmeans.fit_predict(best_vals)

data['CLUSTER'] = y_pred
best_cols.append('CLUSTER')
sbn.pairplot(data[best_cols], hue='CLUSTER')                                    # Plot correlations
plt.show()
