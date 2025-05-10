# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 10:51:56 2024

@author: HP
"""

#''''customer segmentation using K means''''


import pandas as pd 
import numpy as np 
#Load  the dataset
df = pd.read_csv(r"C:\Users\HP\Downloads\Mall_Customers.csv")
# 1
#Check the shape of the dataset 
print('Shape of the dataset:',df.shape)

#check the data types of the columns 
print('\ndata types of the columns:')
print( '\ntypes of the dataset',df.dtypes)
#check for missing values
print('\nMissing values in each columns:')
print(df.isnull().sum())
#get a summary of the numerical columns
df.describe().astype(int)

# Customer Feature Distributions 


import matplotlib.pyplot as plt
import seaborn as sns

# Set the style of the seaborn plot
sns.set(style='whitegrid')

# Create a figure and axis objects
fig, axs = plt.subplots(1, 3, figsize=(20, 5))

# Plot the distribution of age, annual income, and spending score
sns.histplot(data=df, x='Age', kde=True, color='blue', ax=axs[0])
sns.histplot(data=df, x='Annual Income (k$)', kde=True, color='green', ax=axs[1])
sns.histplot(data=df, x='Spending Score (1-100)', kde=True, color='red', ax=axs[2])

# Set the titles of the plots
axs[0].set_title('Age Distribution')
axs[1].set_title('Annual Income Distribution')
axs[2].set_title('Spending Score Distribution')

# Set the title for the entire plot
fig.suptitle('Distribution Analysis of Age, Annual Income, and Spending Score')

# Display the plots
plt.show()
  

#Elbow Method 

from sklearn.cluster import KMeans

#Select the features to use for clustering 
features = df[['Age', 'Annual Income (k$)','Spending Score (1-100)']]

#Determine the optimal number of clusters using the elbow method 
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++',
                    max_iter=300,n_init=10,random_state=0)
    kmeans.fit(features)
    wcss.append(kmeans.inertia_)
    
    #Plot the WCSS values
    plt.plot(range(1,11),wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

#kmeans modeling


#create the kmeans model with the optimal number of(assumed to be 5 based on the elbow method )
kmeans = KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10,random_state=0)
#fit the model to the data and predict the cluster labels
labels = kmeans.fit_predict(features)

#Add the cluster labels to the datAaframe
df['Cluster'] = labels

#display the first few rows of the dataframe with the cluster labels
df.head(100)


#Understanding the cluster characteristics by means values of age,annual income and spending score(1-100) for each cluster

# Calculate the mean values of Age, Annual Income, and Score for each cluster
cluster_means = df.groupby('Cluster')[
['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean().astype(int)

# Display the cluster means
cluster_means
 

#Annual Income VS Spending Score

#Create a scatter plot 

plt.figure(figsize=(10,8))
sns.scatterplot(data=df, x='Annual Income (k$)',y='Spending Score (1-100)',hue='Cluster',palette='viridis', s=100)
#add a title to the plot
plt.title ('Clusters of customers')

#Display the plot
plt.show()
          

'''   Interpreting the Clusters
The clusters we’ve identified provide a clear picture of different customer segments based on their income and spending habits. Here’s a simplified breakdown:

Low income, low spending: Customers with an annual income between 20 and 40k and a spending score between 0 and 40. They have lower incomes and tend to spend less.

High income, low spending: Customers with an annual income between 55 and 140k and a spending score between 0 and 40. Despite their higher incomes, they tend to spend less.

Medium income, medium spending: Customers with an annual income between 40 and 80k and a spending score between 40 and 60. These customers have moderate incomes and spending habits. This is the most concentrated cluster, indicating a large number of customers fall into this category.

Low income, high spending: Customers with an annual income between 20 and 40k and a spending score between 60 and 100. Interestingly, these customers have lower incomes but tend to spend more.

High income, high spending: Customers with an annual income between 70 and 140k and a spending score between 60 and 100. These customers have higher incomes and also tend to spend more.'''

#Customer Segments Based on Age and Spending Score
#This scatter plot provides a visual representation of customer segments based on age and spending score. Each color represents a different cluster, or customer segment
    

#create a scatter plot of age vs spending Score 
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df,x='Age', y='Spending Score (1-100)',hue='Cluster',palette='viridis',s=100)

#add a title to the plot
plt.title('Cluster of customers based on Age and Spending Score')

#Display the plot
plt.show()


'''YOung ,high spenders
20s to 30s youngster are  willing to spend more in spending score 60 to 100

middle age ,moderate spender
20s tp 70s are willin to spend (40 to 60)spend score 

'''

#Gender Distribution Across Clusters

#Create a count plot to show the distribution of 'Gender' within each cluster

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Cluster', hue='Gender')


#add a title to the plot

plt.title('Disribution of Gender within each Cluster')

#display the plot
plt.show()


''' 
in cluster 0,1,2,3 female spend very much by comparision with male 
but  in cluster 4 ,male spend more from females 
'''

#Exploring Pairwise Relationships: A Pairplot Overview
#Each scatterplot in the grid represents the relationship between two features, with the color of the points indicating the cluster to which each customer belongs

#Create a pairplot for relationships between the diffrent features
sns.pairplot(df,hue='Cluster',palette='viridis')

#Display the plot
plt.show()


#Cluster Characteristics :Boxplot Analysis Boxplots are a great way to visualize the distribution of numerical data and can orovide a lot of insights aboutthe characteristics of each cluster
#Create a bocplot for each feature split by 'cluster' 

fig,axs = plt.subplots(1,3, figsize=(20,5))

#boxplot for 'Age'
sns.boxplot(data=df, x='Cluster', y='Age', ax=axs[0])
axs[0].set_title('Age')



#Boxplot for 'Annual Income (k$)'

sns.boxplot(data=df, x='Cluster', y='Annual Income (k$)', ax=axs[1])
axs[1].set_title('Annual Income (k$')

#boxplot for 'Spending Score (1-100)
sns.boxplot(data=df, x='Cluster', y='Spending Score (1-100)', ax=axs[2])
axs[2].set_title('Spending Score (1-100)')

#Display the plots
plt.show()



#Silhouette Score 

#The silhouette score is a measure of how well each datapoint lies within its cluster. 
#It’s a way to quantify the quality of the clustering. The score ranges from -1 to 1.



#Import the necessary libraries
from sklearn.metrics import silhouette_score


#Compute the silhoutte score for the custering
silhouette_score = silhouette_score(df[['Annual Income (k$)','Spending Score (1-100)','Age']],df['Cluster'])
#Print the silhouette score 
print('Silhouette Score: ',silhouette_score)


#Silhouette Score:  0.44428597560893024


#Feature Engineering: ,Different Clustering Algorithm: , Tune KMeans Parameters ,Scaling Features ,Increase the Number of Clusters


#Scaling Feature

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

#Create a StandardScaler object 
scaler = StandardScaler()

#Fit the scaler to the features ands transform them 
scaled_features = scaler.fit_transform(df[['Annual Income (k$)','Spending Score (1-100)','Age']])

# Create a new KMeans object 
kmeans = KMeans(n_clusters=5,random_state=42)

#fit the KMeans object to the scaled features 
kmeans.fit(scaled_features)

#Assign the cluster labels to a new column in the dataframe 
df['Scaled_Cluster'] = kmeans.labels_

#Compute the silhouette score for the new clustering 
scaled_silhouette_score = silhouette_score(scaled_features,df['Scaled_Cluster'])


#print the silhouette score 
print("Scaled Silhouette Score:",scaled_silhouette_score)


#Scaled Silhouette Score: 0.40846873777345605



#Agglomerative Clustering to the scaled features

#Import the necessary libraries 
from sklearn.cluster import AgglomerativeClustering 


#create an AgglomerativeClustering object to the scaled features

agg_clustering = AgglomerativeClustering(n_clusters=5)

#Fit the AgglomerativeClustering object to the scaled features

agg_clustering.fit(scaled_features)

#assign the cluster labels to a new column in the dataframe 
df['Agg_Cluster'] = agg_clustering.labels_

#compute the silhouette score  for the new clustering 
agg_silhouette_score = silhouette_score(scaled_features,df['Agg_Cluster'])

#print the silhouette score 
print("Agglomerative Clustering Silhouette Score: ",agg_silhouette_score)

#Agglomerative Clustering Silhoutte Score

#Agglomerative Clustering Silhouette Score:  0.39002826186267214

#Principal Component Analysis
''' reduce the dimensionality of data. 
It works by finding a new set of dimensions 
(or “principal components”) 
that capture the most variance in the data.'''



from sklearn.decomposition import PCA
import numpy as np

#create a PCA object with two components
pca = PCA(n_components=2)
#fitting the pca to the scaled features and transforming them 
pca_features = pca.fit_transform(scaled_features)


#Creating a NEw KMeans object 
kmeans = KMeans(n_clusters=5,random_state=42)

#Fitting the  KMeans to the PCA features
kmeans.fit(pca_features)

#Assigning the cluster labels to a new column in the dataframe 
df['PCA_Cluster'] = kmeans.labels_

#Calculating the silhouette score for the new clustering 
pca_silhouette_score = silhouette_score(pca_features, df['PCA_Cluster'])

#printing the silhouette score
print('PCA Silhouette Score: ',pca_silhouette_score)

#PCA Silhouette Score:  0.3897861696827718

#DBSCAN (Density-Based Spatial Clustering)
'''
DBSCAN can find arbitrary shaped clusters and
 can identify outliers, which can be an advantage
 over centroid-based clustering algorithms like KMeans.
 '''
 
 #Importing the necessary Libraries
from sklearn.cluster import DBSCAN

#Creating a DBSCAN object
dbscan = DBSCAN(eps=0.5, min_samples=5)

#Fitting the DBSCAN object to the PCA featuresv
dbscan.fit(pca_features)

#Assigning the cluster labels to a new column in the dataframe   

df['DBSCAN_Cluster'] = dbscan.labels_


# Calculating the silhouette score for the new clustering
dbscan_silhouette_score = silhouette_score(pca_features, df['DBSCAN_Cluster'])
#Printing the silhouette score
print('DBSCAN Silhouette Score: ',dbscan_silhouette_score)

#DBSCAN Silhouette Score:  0.3203118288175694 


'''This is indeed lower than the silhouette score 
 obtained from the previous KMeans and
 Agglomerative clustering.
 A lower silhouette score indicates that
 the clusters are less dense and/or
 less well separated than in the previous
 clustering solutions.
 '''
 
 
 # Interpretation and Insights 
 # Grouping the data by 'Cluster' and calculating the mean of the original features
cluster_characteristics = df.groupby('Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean().astype(int)

# Displaying the characteristics of each cluster
print(cluster_characteristics)

#Cluster 0: “Conservative Middle-Aged” --This group might be less \
    ##responsive to marketing efforts aimed at increasing spending.

#CLUSTER 1: “Young High Earners”--This group could be a key target for marketing campaigns,
#as they have both the means and the willingness to spend.
    

#Cluster 2: “Balanced Middle-Aged” —Marketing strategies for this group might need to focus on value and quality,
# as they might be more discerning with their spending.

#Cluster 3: “Wealthy Savers”--They might value saving, or they might
# not see value in the current offerings.

#Cluster 4: “Young Spenders” —This cluster includes young individuals
# (average age 25) with low annual income (average $26k), 
#but their spending score is high (average 78). This suggests that these individuals, despite having lower income, are willing to spend a significant portion of it.
# They might be impulse buyers, or they might value experiences or products over saving money.
'''
Conclusion

The results of this project can provide valuable insights for the marketing team. 
For example, they can target their marketing campaigns to the different segments of customers, based on
 their characteristics. They can also use this information to develop new products or services that cater
 to the specific needs and preferences of each segment.
 '''
 