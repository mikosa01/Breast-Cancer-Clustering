# Breast-Cancer-Clustering

Breast cancer is an illness that originates in the breast tissue. Symptoms of breast cancer include a lump within the breast, a change in breast form, dimpling in the skin milk rejection, fluid seeping from the nipple, and many more. Patients suffering from distant infection spread may have bone pain, enlarged lymph nodes, a shortness of breath, or yellow skin. According to World Health Data, breast cancer injured 2.3 million women worldwide, killing 685 000 people. Since the end of 2020, 7.8 million women have been diagnosed with breast cancer in the past five years, making it the world's most common malignancy. Breast cancer is the leading cause of disability-adjusted life years (DALYs) in women worldwide. Breast cancer affects women anywhere at age after puberty in any nation, with the acquire additional with age..
From the 1930s until the 1970s, the death rate from breast cancer changed just slightly. In nations where early detection programmes are paired with different types of therapy to eradicate invasive sickness, survival rates began to climb in the 1980s. The dataset used for this report was obtained in the year 1995-11-01 after a diagnostic test was carried out on patients who suffered from breast cancer in Wisconsin. According to the analysis been carried out on the dataset. 
 
In order to know the accuracy of my model, the following metrics were used:
1) Normalized Mutual Information (NIM): It scales its results between 0 and 1. This function is normalized by generalizing the mean of the true label and predicted label which is defined by the average method. 
2) Rand index adjusted: It calculates a similarity measure across two clustering by taking into account all sample combinations and counting pairings allocated to the same or different clusters in the anticipated and true clustering.

#Graph Base Clustering


Graph-based clustering is a sort of clustering approach that converts data into a graphical representation in which the vertices of the data are the clustering of the data points and the weighting of the edges based on the similarity of the vertices. It includes clustering techniques such as Chameleon algorithms, Spectral algorithms, Minimum Spanning Tree algorithms, and many more; however, my focus will be on Spectral clustering, which is the algorithm used for this research. As a prelude to discussing the spectral algorithm, I would like to discuss Sparsification, a strategy used by all graph-based algorithms.
Sparsification is the process of reducing or breaking those connections with a similarity under a certain threshold. This is accomplished by retaining just links that are the point's k nearest neighbours.. Following are the ways in which this method has been effective.
1) Data reduction : The amount of data required to cluster is drastically reduced. Using a data proximity matrix, it can also reduce data by about 99%.
2) Clustering works better. It keeps the connection to the nearest neighbor and also breaks connections to more distant objects. It reduces noise, outliers and sharpens the distinctness between clusters.
3) Due to the sparsification of such proximity graph, graph partitioning algorithms such as Opossum and Chameleon can be utilized for the clustering step.

#Spectral clustering

Spectral clustering is a graph theory-based approach for discovering communities of nodes in a network based on the links that connect them. Assuming no assumptions about the cluster shape, it works well with intertwined shapes and utilizes an iterative process to find local minima. The spectrum of the matrices representation  is analysed by spectral graph theory. A spectrum is a set of eigenvectors ranked according to the magnitude of their corresponding eigenvalues. The eigenvalues of the adjacency matrix can be used to draw a graph. The spectrum properties of a graph are related to the graph partitioning challenge. Firstly, we have to create a graph's similarity/adjacency matrix, W and then create a diagonal matrix D.

![formula](https://user-images.githubusercontent.com/41128084/214036576-7cc23451-58a6-4370-9838-1146e0c5bebd.PNG)

It is worth mentioning that Dij signifies the degree of node i when W is a binary 0/1 matrix.
L=D-W 

Consider a data set that has N data points.
1. Construct an N' N similarity matrix, W
2. Determine the N'N Laplacian matrices, L = D - W.
3. Find the k "smallest" L eigenvectors.
a) Each eigenvector vi is a column vector of size N'1.
b) Create a matrix V with the columns eigenvectors v1, v2,..., vk.
4. Using k-means or another clustering technique, divide the rows in V into K groups.



