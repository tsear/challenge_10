#!/usr/bin/env python
# coding: utf-8

# # Module 10 Application
# 
# ## Challenge: Crypto Clustering
# 
# In this Challenge, you’ll combine your financial Python programming skills with the new unsupervised learning skills that you acquired in this module.
# 
# The CSV file provided for this challenge contains price change data of cryptocurrencies in different periods.
# 
# The steps for this challenge are broken out into the following sections:
# 
# * Import the Data (provided in the starter code)
# * Prepare the Data (provided in the starter code)
# * Find the Best Value for `k` Using the Original Data
# * Cluster Cryptocurrencies with K-means Using the Original Data
# * Optimize Clusters with Principal Component Analysis
# * Find the Best Value for `k` Using the PCA Data
# * Cluster the Cryptocurrencies with K-means Using the PCA Data
# * Visualize and Compare the Results

# ### Import the Data
# 
# This section imports the data into a new DataFrame. It follows these steps:
# 
# 1. Read  the “crypto_market_data.csv” file from the Resources folder into a DataFrame, and use `index_col="coin_id"` to set the cryptocurrency name as the index. Review the DataFrame.
# 
# 2. Generate the summary statistics, and use HvPlot to visualize your data to observe what your DataFrame contains.
# 
# 
# > **Rewind:** The [Pandas`describe()`function](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html) generates summary statistics for a DataFrame. 

# In[1]:


# Import required libraries and dependencies
import pandas as pd
import hvplot.pandas
from path import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# In[2]:


# Load the data into a Pandas DataFrame
df_market_data = pd.read_csv(
    Path("Resources/crypto_market_data.csv"),
    index_col="coin_id")

# Display sample data
df_market_data.head(10)


# In[3]:


# Generate summary statistics
df_market_data.describe()


# In[4]:


# Plot your data to see what's in your DataFrame
df_market_data.hvplot.line(
    width=800,
    height=400,
    rot=90
)


# ---

# ### Prepare the Data
# 
# This section prepares the data before running the K-Means algorithm. It follows these steps:
# 
# 1. Use the `StandardScaler` module from scikit-learn to normalize the CSV file data. This will require you to utilize the `fit_transform` function.
# 
# 2. Create a DataFrame that contains the scaled data. Be sure to set the `coin_id` index from the original DataFrame as the index for the new DataFrame. Review the resulting DataFrame.
# 

# In[5]:


# Use the `StandardScaler()` module from scikit-learn to normalize the data from the CSV file
scaled_data = StandardScaler().fit_transform(df_market_data)


# In[6]:


# Create a DataFrame with the scaled data
df_market_data_scaled = pd.DataFrame(
    scaled_data,
    columns=df_market_data.columns
)

# Copy the crypto names from the original data
df_market_data_scaled["coin_id"] = df_market_data.index

# Set the coinid column as index
df_market_data_scaled = df_market_data_scaled.set_index("coin_id")

# Display sample data
df_market_data_scaled.head()


# ---

# ### Find the Best Value for k Using the Original Data
# 
# In this section, you will use the elbow method to find the best value for `k`.
# 
# 1. Code the elbow method algorithm to find the best value for `k`. Use a range from 1 to 11. 
# 
# 2. Plot a line chart with all the inertia values computed with the different values of `k` to visually identify the optimal value for `k`.
# 
# 3. Answer the following question: What is the best value for `k`?

# In[7]:


# Create a list with the number of k-values to try
# Use a range from 1 to 11
k = list(range(1, 11))


# In[8]:


# Create an empy list to store the inertia values
inertia = []


# In[9]:


# Create a for loop to compute the inertia with each possible value of k
# Inside the loop:

for i in k:
    model=KMeans(n_clusters=i, random_state=0)
    model.fit(df_market_data_scaled)
    inertia.append(model.inertia_)


# In[10]:


# Create a dictionary with the data to plot the Elbow curve
elbow_data = {
    'k': k,
    'inertia': inertia
}

# Create a DataFrame with the data to plot the Elbow curve
df_elbow = pd.DataFrame(elbow_data)


# In[52]:


# Plot a line chart with all the inertia values computed with 
# the different values of k to visually identify the optimal value for k.
df_elbow.hvplot.line(
    x='k',
    y='inertia',
    title='Elbow',
    xticks=k
)


# In[53]:


plot_original_elbow = df_elbow.hvplot.line(
    x='k',
    y='inertia',
    title='Elbow',
    xticks=k
)


# #### Answer the following question: What is the best value for k?
# **Question:** What is the best value for `k`?
# 
# **Answer:** 4

# ---

# ### Cluster Cryptocurrencies with K-means Using the Original Data
# 
# In this section, you will use the K-Means algorithm with the best value for `k` found in the previous section to cluster the cryptocurrencies according to the price changes of cryptocurrencies provided.
# 
# 1. Initialize the K-Means model with four clusters using the best value for `k`. 
# 
# 2. Fit the K-Means model using the original data.
# 
# 3. Predict the clusters to group the cryptocurrencies using the original data. View the resulting array of cluster values.
# 
# 4. Create a copy of the original data and add a new column with the predicted clusters.
# 
# 5. Create a scatter plot using hvPlot by setting `x="price_change_percentage_24h"` and `y="price_change_percentage_7d"`. Color the graph points with the labels found using K-Means and add the crypto name in the `hover_cols` parameter to identify the cryptocurrency represented by each data point.

# In[12]:


# Initialize the K-Means model using the best value for k
model = KMeans(n_clusters=4)


# In[13]:


# Fit the K-Means model using the scaled data
model.fit(df_market_data_scaled)


# In[14]:


# Predict the clusters to group the cryptocurrencies using the scaled data
clusters = model.predict(df_market_data_scaled)

# View the resulting array of cluster values.
clusters


# In[15]:


# Create a copy of the DataFrame
df_predictions = df_market_data_scaled.copy()


# In[16]:


# Add a new column to the DataFrame with the predicted clusters
df_predictions['Clusters'] = clusters

# Display sample data
df_predictions.head()


# In[44]:


# Create a scatter plot using hvPlot by setting 
# `x="price_change_percentage_24h"` and `y="price_change_percentage_7d"`. 
# Color the graph points with the labels found using K-Means and 
# add the crypto name in the `hover_cols` parameter to identify 
# the cryptocurrency represented by each data point.

df_predictions.hvplot.scatter(
    x='price_change_percentage_24h',
    y='price_change_percentage_7d',
    by='Clusters',
    hover_cols='coin_id',
    title='Crypto Clusters'
)


# In[45]:


plot_original_data = df_predictions.hvplot.scatter(
    x='price_change_percentage_24h',
    y='price_change_percentage_7d',
    by='Clusters',
    hover_cols='coin_id',
    title='Crypto Clusters'
)


# ---

# ### Optimize Clusters with Principal Component Analysis
# 
# In this section, you will perform a principal component analysis (PCA) and reduce the features to three principal components.
# 
# 1. Create a PCA model instance and set `n_components=3`.
# 
# 2. Use the PCA model to reduce to three principal components. View the first five rows of the DataFrame. 
# 
# 3. Retrieve the explained variance to determine how much information can be attributed to each principal component.
# 
# 4. Answer the following question: What is the total explained variance of the three principal components?
# 
# 5. Create a new DataFrame with the PCA data. Be sure to set the `coin_id` index from the original DataFrame as the index for the new DataFrame. Review the resulting DataFrame.

# In[18]:


# Create a PCA model instance and set `n_components=3`.
pca = PCA(n_components=3)


# In[19]:


# Use the PCA model with `fit_transform` to reduce to 
# three principal components.
pca_data = pca.fit_transform(df_market_data_scaled)

# View the first five rows of the DataFrame. 
pca_data


# In[20]:


# Retrieve the explained variance to determine how much information 
# can be attributed to each principal component.
pca.explained_variance_ratio_


# #### Answer the following question: What is the total explained variance of the three principal components?
# 
# **Question:** What is the total explained variance of the three principal components?
# 
# **Answer:** the first component represents 37% of the variance, followed by 35% & 18%, respectively.

# In[21]:


# Create a new DataFrame with the PCA data.
df_crypto_pca = pd.DataFrame(pca_data, columns=["PC1", "PC2", "PC3"])
    

# Copy the crypto names from the original data
df_crypto_pca["coin_id"] = df_market_data_scaled.index

# Set the coinid column as index
df_crypto_pca = df_crypto_pca.set_index("coin_id")

# Display sample data
df_crypto_pca.head()


# ---

# ### Find the Best Value for k Using the PCA Data
# 
# In this section, you will use the elbow method to find the best value for `k` using the PCA data.
# 
# 1. Code the elbow method algorithm and use the PCA data to find the best value for `k`. Use a range from 1 to 11. 
# 
# 2. Plot a line chart with all the inertia values computed with the different values of `k` to visually identify the optimal value for `k`.
# 
# 3. Answer the following questions: What is the best value for k when using the PCA data? Does it differ from the best k value found using the original data?

# In[22]:


# Create a list with the number of k-values to try
# Use a range from 1 to 11
k = list(range(1, 11))


# In[23]:


# Create an empy list to store the inertia values
inertia = []


# In[24]:


# Create a for loop to compute the inertia with each possible value of k
# Inside the loop:
# 1. Create a KMeans model using the loop counter for the n_clusters
# 2. Fit the model to the data using `df_market_data_pca`
# 3. Append the model.inertia_ to the inertia list

for i in k:
    model = KMeans(n_clusters=i, random_state=0)
    model.fit(df_crypto_pca)
    inertia.append(model.inertia_)


# In[25]:


# Create a dictionary with the data to plot the Elbow curve
elbow_data = {
    'k': k,
    'inertia': inertia
}

# Create a DataFrame with the data to plot the Elbow curve
df_elbow = pd.DataFrame(elbow_data)


# In[48]:


# Plot a line chart with all the inertia values computed with 
# the different values of k to visually identify the optimal value for k.
df_elbow.hvplot.line(
    x='k',
    y='inertia',
    xticks=k
)


# In[50]:


plot_crptyo_elbow = df_elbow.hvplot.line(
    x='k',
    y='inertia',
    xticks=k
)


# #### Answer the following questions: What is the best value for k when using the PCA data? Does it differ from the best k value found using the original data?
# * **Question:** What is the best value for `k` when using the PCA data?
# 
#   * **Answer:** 4
# 
# 
# * **Question:** Does it differ from the best k value found using the original data?
# 
#   * **Answer:** not in this instance

# ---

# ### Cluster Cryptocurrencies with K-means Using the PCA Data
# 
# In this section, you will use the PCA data and the K-Means algorithm with the best value for `k` found in the previous section to cluster the cryptocurrencies according to the principal components.
# 
# 1. Initialize the K-Means model with four clusters using the best value for `k`. 
# 
# 2. Fit the K-Means model using the PCA data.
# 
# 3. Predict the clusters to group the cryptocurrencies using the PCA data. View the resulting array of cluster values.
# 
# 4. Add a new column to the DataFrame with the PCA data to store the predicted clusters.
# 
# 5. Create a scatter plot using hvPlot by setting `x="PC1"` and `y="PC2"`. Color the graph points with the labels found using K-Means and add the crypto name in the `hover_cols` parameter to identify the cryptocurrency represented by each data point.

# In[27]:


# Initialize the K-Means model using the best value for k
model = KMeans(n_clusters=4)


# In[28]:


# Fit the K-Means model using the PCA data
model.fit(df_crypto_pca)


# In[29]:


# Predict the clusters to group the cryptocurrencies using the PCA data
clusters2 = model.predict(df_crypto_pca)

# View the resulting array of cluster values.
clusters2


# In[30]:


# Create a copy of the DataFrame with the PCA data
df_pca_predictions = df_crypto_pca.copy()

# Add a new column to the DataFrame with the predicted clusters
df_pca_predictions['Clusters'] = clusters2

# Display sample data
df_pca_predictions


# In[41]:


# Create a scatter plot using hvPlot by setting 
# `x="PC1"` and `y="PC2"`. 
# Color the graph points with the labels found using K-Means and 
# add the crypto name in the `hover_cols` parameter to identify 
# the cryptocurrency represented by each data point.

df_pca_predictions.hvplot.scatter(
    x="PC1",
    y="PC2",
    by='Clusters',
    hover_cols=["coin_id"],
    title="Crypto Scatter Plot, PCA data"
)


# In[42]:


crypto_plot = df_pca_predictions.hvplot.scatter(
    x="PC1",
    y="PC2",
    by='Clusters',
    hover_cols=["coin_id"],
    title="Crypto Scatter Plot, PCA data"
)


# ---

# ### Visualize and Compare the Results
# 
# In this section, you will visually analyze the cluster analysis results by contrasting the outcome with and without using the optimization techniques.
# 
# 1. Create a composite plot using hvPlot and the plus (`+`) operator to contrast the Elbow Curve that you created to find the best value for `k` with the original and the PCA data.
# 
# 2. Create a composite plot using hvPlot and the plus (`+`) operator to contrast the cryptocurrencies clusters using the original and the PCA data.
# 
# 3. Answer the following question: After visually analyzing the cluster analysis results, what is the impact of using fewer features to cluster the data using K-Means?
# 
# > **Rewind:** Back in Lesson 3 of Module 6, you learned how to create composite plots. You can look at that lesson to review how to make these plots; also, you can check [the hvPlot documentation](https://holoviz.org/tutorial/Composing_Plots.html).

# In[54]:


# Composite plot to contrast the Elbow curves
plot_crptyo_elbow + plot_original_elbow


# In[46]:


# Compoosite plot to contrast the clusters
plot_original_data + crypto_plot


# #### Answer the following question: After visually analyzing the cluster analysis results, what is the impact of using fewer features to cluster the data using K-Means?
# 
#   * **Question:** After visually analyzing the cluster analysis results, what is the impact of using fewer features to cluster the data using K-Means?
# 
#   * **Answer:** We get a much more accurate look at the clusters having fewer features, thus providing us with more acccurate & actionable data. 
