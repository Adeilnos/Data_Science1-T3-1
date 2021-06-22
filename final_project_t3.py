"""

@author: Leonidas, Tuhera, Kristiyan

This module is a simple web frontend to select algorithms, hyperparameters, datasets and visualize the outcome.
"""



import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn import metrics
import seaborn as sns
from sklearn.cluster import Birch


st.title('Data Science 1 Project group T3-1')


#create selectboxes to feed selextion to algorithm
dataset_str = st.selectbox('Choose dataset (mandatory)',["Aggregation.txt","jain.txt","s1.txt"], key="1")

algorithm_str = st.selectbox('Choose algorithm  (mandatory)',["birch","kmeans","dbscan"], key="9")

clusters = st.selectbox('Choose number of clusters  (mandatory)',list(range(2,999)), key="2")

st.write("Adjust K-Means Hyperparameters (optional)")

ITERATIONS = st.selectbox('Choose number of iterations for K-Means',list(range(1,999)), key="3")

n_init = st.selectbox('Choose number of n_init for K-Means',list(range(1,999)), key="4")

st.write("Adjust BIRCH Hyperparameters (optional)")

branching_factor = st.selectbox('Choose branching_factor for BIRCH',list(range(1,999)), key="5")

threshold = st.selectbox('Choose threshold for BIRCH',list(range(1,999)), key="6")

st.write("Adjust DBSCAN Hyperparameters (optional)")

eps = st.selectbox('Choose eps for DBSCAN',list(range(1,999)), key="7")

min_samples = st.selectbox('Choose min_samples for DBSCAN',list(range(1,999)), key="8")

run = st.button('Run', key="10")


if algorithm_str == "kmeans" and run:


    def kmeans(clusters, ITERATIONS, n_init,dataset_str):
    #read and structure dataset
        input_data = np.loadtxt(dataset_str)
        true_labels = np.loadtxt('s1-label.pa')
        
        #other data preprocesing
        if dataset_str != "s1.txt":
            #get labels from third row
            true_labels = input_data[:,2]
            #delete label row
            input_data = input_data[:,:2]
        
        #x and y datapoints
        x_points = input_data[:,0]
        y_points = input_data[:,1]
        
                
        #random centroid initilization. Choose the best centroid of n_init rounds.
        km = KMeans(n_clusters=clusters,init='random',max_iter=ITERATIONS,n_init=n_init)
        km.fit(input_data)
        
        #Get classified labels
        labels = np.array(km.labels_)
        #Get centroids
        centroids = km.cluster_centers_
        #Plot the dataset (all points in blue)
        plt.scatter(x_points,y_points,c="blue")
        #Plot the centroids after the number of iterations
        plt.scatter(centroids[:,0],centroids[:,1],c=list(range(0, clusters)))
        plt.title(f'Data with centroids (after {ITERATIONS} iterations)')
        plt.show()
        
        #Clear plot
        plt.clf()
        
        #Plot the datapoints again, now with the corresponding color for each label.
        fig, ax = plt.subplots()
        plt.scatter(x_points,y_points,c=labels)
        plt.title(f'Classification using the centroids after {ITERATIONS} iterations')
        st.write(plt.show())
        st.pyplot(fig) 
        
        #evaluation metrics: calc silhouette and purity score
        silhouette = round(silhouette_score(input_data, labels),3)
        
        st.write(f'Silhouette Score(n={clusters}): {silhouette}')
        
        def purity_score(true, pred):
            # compute confusion / contingency matrix
            contingency_matrix = metrics.cluster.contingency_matrix(true, pred)
            # calculate purity as return
            return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
        
        
        purity = round(purity_score(true_labels, labels),3)
        
        st.write(f'purity Score(n={clusters}): {purity}')
    
    #run method with chosen parameters
    kmeans(clusters, ITERATIONS, n_init,dataset_str)
    
elif algorithm_str == "dbscan" and run:
    
    def dbscan(clusters, eps, min_samples,dataset_str):
        
        # Importing the dataset
        #dataset = pd.read_csv('s1.txt')
        dataset = np.loadtxt(dataset_str)
        
        if dataset_str == "s1.txt":
            true_labels = np.loadtxt('s1-label.pa')
        else:
            true_labels = dataset[:,2]
            
        
        #print(dataset.head())
        X = dataset[:,:2]
        if dataset_str == "s1.txt":
            X = dataset
        fig = plt.figure(figsize=(10, 10))
        
        
        # Using the elbow method to find the optimal number of clusters
        from sklearn.cluster import DBSCAN
        dbscan=DBSCAN(eps=eps,min_samples=min_samples)
        
        # Fitting the model
        
        model=dbscan.fit(X)
        
        labels=model.labels_
        sns.scatterplot(X[:,0], X[:,1], hue=["cluster-{}".format(x) for x in labels])
        st.pyplot(fig)
        
        #from sklearn import metrics
        
        #identifying the points which makes up our core points
        sample_cores=np.zeros_like(labels,dtype=bool)
        
        sample_cores[dbscan.core_sample_indices_]=True
        
        #Calculating the number of clusters
        
        n_clusters=len(set(labels))- (1 if -1 in labels else 0)
        
        
        
        st.write("The silhouette score is:",metrics.silhouette_score(X,labels))
        st.write("number of found clusters:", n_clusters)
        
    #run method with chosen parameters
    dbscan(clusters, eps, min_samples,dataset_str)


elif algorithm_str == "birch" and run:
    
    def birch(clusters, branching_factor, threshold,dataset_str):
        #dataset = pd.read_csv("lp1.csv")
        dataset = np.loadtxt(dataset_str)
        if dataset_str == "s1.txt":
            true_labels = np.loadtxt('s1-label.pa')
        else:
            true_labels = dataset[:,2]

        
        X = dataset[:,:2]
        if dataset_str == "s1.txt":
            X = dataset
        
        #print(dataset.head())
        
        
        
        # Creating the BIRCH clustering model
        model = Birch(branching_factor = branching_factor, n_clusters=clusters, threshold =threshold)
        
        
        # Fit the data (Training)
        model.fit(X)
        labels=model.labels_
        # Predict the same data
        pred = model.predict(X)
        
        # Creating a scatter plot
        fig, ax = plt.subplots()
        plt.scatter(dataset[:, 0], dataset[:, 1], c = pred, cmap = 'rainbow',
                    alpha = 0.7, edgecolors = 'b')
        st.pyplot(fig)# Import required libraries and modules
        #print(metrics.silhouette_score(X,labels))
        st.write("The silhouette score is:", metrics.silhouette_score(X,labels))
        
    #run method with chosen parameters    
    birch(clusters, branching_factor, threshold,dataset_str)

