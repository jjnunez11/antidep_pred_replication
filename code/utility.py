# -*- coding: utf-8 -*-
"""
Helper functions used for other parts of the code
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.feature_selection import chi2
from sklearn.linear_model import ElasticNetCV
from sklearn.cluster import FeatureAgglomeration
from sklearn.metrics import roc_curve, auc
from astropy.stats import bootstrap as astro_bootstrap
import matplotlib.pyplot as plt

##Helper Functions -

def bootstrap(X,n):
    """ Produce one boostrap sample from X of n example. Uses astropy's boostrap """
    bootsample = astro_bootstrap(X,bootnum=1,samples=n)
    return bootsample[0,:,:]

# subsample
# Input: X n,d data matrix, i int

def subsample(X,t=30) :
    """ Produces subsamples as from the paper, using boostraps from the majority class
    (the class with more examples) and all of the minority class divided into t subsamples
    Arguments:
        X -- Data matrix of n,d dimensions
        t -- How many subsamples to produce; in the Zie paper, they used 30
    Returns:
        t copys of training data and labels
    """
    n, d = X.shape

    count_1 = np.sum(X[:,0] == 1)
    count_0 = np.sum(X[:,0] == 0)

    # Set majority and minority classes
    if count_1 > count_0:
        major = 1
        minor = 0
        count_minor = count_0
    else:
        major = 0
        minor = 1
        count_minor = count_1

    # throw exception if there are data entries not 1 or 0 in this column
    assert ((count_1 + count_0) == n), "Column to decide classes has non 0 or 1 entries; must be binary"

    # Create matrices of only majority and minority examples
    X_major = X[X[:,0] == major,:] # Samples with values of the majority in coloumn c
    X_minor = X[X[:,0] == minor,:] # "" but minority

    # Loop to make the t subsamples
    trains = np.zeros((t,2*count_minor,d-1))
    labels = np.zeros((t,2*count_minor))
    for i in range(t):
        #Create empty array
        subsample = np.zeros((2*count_minor,d))
        # Add into boostraps from majority examples
        subsample[0:count_minor,:] = bootstrap(X_major,count_minor)
        # Add in examples from minority examples (not bootstrapped)
        subsample[count_minor:,:] = X_minor
        # Shuffle data in case this matters
        np.random.shuffle(subsample)
        trains[i,:,:] = subsample[:,1:]
        labels[i] = subsample[:,0]

    return trains, labels

#print("Here's an example of using subsample, it has shape: " + str(subsample(X,0,10).shape))

def featureSelectionChi(X,y,n,k):
    """ ChiSquareMethod to select features
    Arguments:
        X: Training set
        y: Training label
        k: Number of clusters
        n: Top n features
    """

    #Using the centroid of k-means clustering to represent a cluster of features
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X.T)
    features = pairwise_distances_argmin(kmeans.cluster_centers_, X.T)
    Xnew = X[:,features]
    F, _ = chi2(Xnew, y)
    #selecting top n feature
    seed = np.argsort(-F)
    topFeatures = features[seed[:n]]
    #print(topFeatures)
    return topFeatures

def featureSelectionELAS(X,y,n):
    """ Elastic net method to select features
    Arguments:
        X: Training set
        y: Training label
        n: Top n features
    """
    # Use cross validation to selection best elasticnet model first
    cv_model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9], eps=0.005, n_alphas=30, fit_intercept=True,
                        normalize=True, precompute='auto', max_iter=300, tol=0.0001, cv=5,
                        copy_X=True, verbose=0, n_jobs=-1, positive=False, random_state=None)
    cv_model.fit(X, y)

    # Select best features according to absolute value of coefficients
    features = np.argsort(-np.abs(cv_model.coef_))
    #print(features[:n])
    return features[:n]

def featureSelectionAgglo(X,k):
    """ Feature aggolomeraion to select features
    Arguments
        X: The first column of X are labels
        k: Number of clusters of features
    """
    agglo = FeatureAgglomeration(n_clusters=k)
    agglo.fit(X)
    labelCluster = agglo.labels_[0]
    features = np.where(agglo.labels_==labelCluster)[0][1:]-1
    n = features.size
    return features, n

def drawROC(y_true,y_score):
    """ Function to draw ROC curve
    Arguments:
        y_true: True label
        y_score: Probability for positive label
    """
    fpr, tpr, thresholds = roc_curve(y_true,y_score)
    plt.title("ROC curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot(fpr, tpr)
    print("AUC:", auc(fpr, tpr))
    return auc(fpr, tpr)

