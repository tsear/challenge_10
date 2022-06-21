# challenge_10

This challenge asked us to use unsupervised learning &, specifically, the KMeans algorithim from scikitlearn to build a cluster of datapoints based on crypto prices.

The first portion of the app is reading in the data from resources/csv, cleaning, normalizing, fitting, & indexing the data so we can take proper action on it with our model. 

The second portion of the assignment asks us to use the Elbow method to find the perfect k value for our model, then implimenting that variable into our first iteration of the mdoel using the scaled data as its training. 

The third portion of the assignment asks for a similar creation, but this time using PCA instead of the scaled data. This reduces the ammount of features to be used by the application & allows it to get a more concise picture of the clusters, providing more accurate data than the scaled data, at large. 

Finally, we are asked to composite plot all four of our output graphs, & determine which method was the best course of action. 
