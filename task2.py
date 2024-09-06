import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

#load the dataset
df=pd.read_csv('C:\\Users\\nivas\\OneDrive\\Pictures\\Documents\\ML SkillCraft\\Mall_Customers.csv')

#check the shape of your dataset
print('Shape of the dataset:',df.shape)

#check the datatype of the columns
print(df.dtypes)

#check for missing values
print('\nMissing values in each column:')
print(df.isnull().sum())

#get summary of the numerical columns
df.describe().astype(int)

#set the style of the seaborn plot
sns.set(style='whitegrid')

#create a figure and axis bjects
fig, axs=plt.subplots(1,3,figsize=(20,5))

#plot the distribution of age,annual incme, and spending score
sns.histplot(data=df,x='Age',kde=True,color='blue',ax=axs[0])
sns.histplot(data=df, x='Annual Income (k$)', kde=True, color='green', ax=axs[1])
sns.histplot(data=df, x='Spending Score (1-100)', kde=True, color='red', ax=axs[2])

#set the titles of the plot
axs[0].set_title('Age Distribution')
axs[1].set_title('Annual Income Distribution')
axs[2].set_title('Spending Score Distribution')

#set the title for the entire plot
fig.suptitle('Distribution Analysis of Age, Annual Income, and Spending Score')

# Display the plots
plt.show()

#select the features to use for clustering
features=df[['Age','Annual Income (k$)','Spending Score (1-100)']]

#determine the optimal number of clusters using the elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(features)
    wcss.append(kmeans.inertia_)
    
#plot the WCSS values
plt.plot(range(1,11),wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()   

# Create the KMeans model with the optimal number of clusters (assumed to be 5 based on the elbow method)
kmeans=KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10,random_state=0)    
#fit the model to the data and predict the cluster labels
labels=kmeans.fit_predict(features)

#add the cluster label to the dataframe
df['Cluster'] =labels
#display the first few rows of the dataframe with the cluster labels
df.head(100)

# Calculate the mean values of Age, Annual Income, and Score for each cluster
cluster_means = df.groupby('Cluster')[
['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean().astype(int)

# Display the cluster means
cluster_means       

# Create a scatter plot of 'Annual Income (k$)' vs 'Spending Score (1-100)' colored by 'Cluster'
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='viridis', s=100)

# Add a title to the plot
plt.title('Clusters of customers')

# Display the plot
plt.show() 

# Create a scatter plot of Age vs Spending Score 
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df, x='Age', y='Spending Score (1-100)', hue='Cluster', 
palette='viridis', s=100)

# Add a title to the plot
plt.title('Clusters of customers based on Age and Spending Score')

# Display the plot
plt.show()    

