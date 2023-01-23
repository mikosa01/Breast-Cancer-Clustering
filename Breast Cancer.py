import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.cluster import KMeans, SpectralClustering
from kneed import KneeLocator
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches
import warnings 

#Load the dataset with pandas
cols = ['Sample_code_number', 'Clump_Thickness', 'Uniformity_of_Cell_Size', 
        'Uniformity_of_Cell_Shape','Marginal_Adhesion', 'Single_Epithelial_Cell_Size',
        'Bare_Nuclei', 'Bland_Chromatin', 'Normal_Nucleoli', 'Mitoses', 'Class']
df = pd.read_csv('breast-cancer.csv', names = cols, header = None)

#drop the unwanted col
df.drop(['Sample_code_number'], 1, inplace = True)
df.head()

#Changing the column with string dtypes to int
df['Bare_Nuclei'] = df['Bare_Nuclei'].replace('?', np.nan)
df['Bare_Nuclei'] = pd.to_numeric(df['Bare_Nuclei'])

#drop all row with missing values 
df.dropna(axis = 0, inplace = True)
label_true = df.Class

#drop the label class
df.drop('Class', 1, inplace = True)

#Scale the dataframe to reduce outliers 
df_scaler = StandardScaler().fit_transform(df)

#Reducing the dimensionality of the dataframe
decomp = PCA(n_components = 3 )
df_pca = decomp.fit_transform(df_scaler)

#Create a scatterplot of new reduced dataframe
plt.scatter(x =df_pca[:,0], y= df_pca[:,1])
plt.show()

#Using kmeans to generate the optimal cluster number
num_cluster = range(1, 11)
sse = []
for x in num_cluster: 
    kmeans = KMeans(n_clusters = x, init = 'k-means++', n_init = 15, 
                    random_state= 45)
    kmeans.fit(df_pca)
    sse.append(kmeans.inertia_)

#Get optimal cluster from the kmeans clustering
kl = KneeLocator(x = num_cluster, y = sse, direction = 'decreasing', 
                 curve = 'convex')
num_clusters = kl.elbow
#print(num_cluster)

#Generate a spectral clustering with the optimal number of cluster 
spec = SpectralClustering(n_clusters = num_clusters, n_init = 25, 
                          random_state = 42, affinity='nearest_neighbors',
                            assign_labels = 'discretize' )

spec.fit_predict(df)
label = spec.labels_
pca_1 = df_pca.copy()
df['Cluster'] = label


#filter rows of original data
filtered_label0 = pca_1[label == 0]
 
filtered_label1 = pca_1[label == 1]
 
#Plotting the results
plt.scatter(filtered_label0[:,0] , filtered_label0[:,1] , color = 'red')
plt.scatter(filtered_label1[:,0] , filtered_label1[:,1] , color = 'black' )
plt.xlabel('X')
plt.ylabel('Y')
labela = mpatches.Patch(color='red', label='Cluster 1')
labelb = mpatches.Patch(color='black', label = 'Cluster 2')
plt.legend(handles = [labela, labelb])
plt.savefig('2d.png')
plt.show()

norm = normalized_mutual_info_score(label_true, label)
print('The normalized mutual information score is : {}'.format(norm))

rand = adjusted_rand_score(label_true, label)
print('The ajusted rand score is : {}'.format(rand))

#Generating a dataframe of the cluster's value count
cluster = df['Cluster'].value_counts(ascending = True).rename_axis('cluster').reset_index(name='counts')

#Mapping th cluster name to cluster 1 and cluster 2 respectively
cluster['cluster'] = cluster['cluster'].map({0:'Cluster 1', 1:'Cluster 2' })

#plotting a barplot of cluster to the data point count.
sns.barplot(x = cluster['cluster'], y = cluster['counts'])
plt.savefig('plot2.png')