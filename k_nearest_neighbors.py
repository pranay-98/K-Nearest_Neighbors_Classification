# k_nearest_neighbors.py: Machine learning implementation of a K-Nearest Neighbors classifier from scratch.
#
# Submitted by: [enter your full name here] -- [enter your IU username here]
#
# Based on skeleton code by CSCI-B 551 Fall 2021 Course Staff

import numpy as np
from utils import euclidean_distance, manhattan_distance
from collections import Counter


class KNearestNeighbors:
    """
    A class representing the machine learning implementation of a K-Nearest Neighbors classifier from scratch.

    Attributes:
        n_neighbors
            An integer representing the number of neighbors a sample is compared with when predicting target class
            values.

        weights
            A string representing the weight function used when predicting target class values. The possible options are
            {'uniform', 'distance'}.

        _X
            A numpy array of shape (n_samples, n_features) representing the input data used when fitting the model and
            predicting target class values.

        _y
            A numpy array of shape (n_samples,) representing the true class values for each sample in the input data
            used when fitting the model and predicting target class values.

        _distance
            An attribute representing which distance metric is used to calculate distances between samples. This is set
            when creating the object to either the euclidean_distance or manhattan_distance functions defined in
            utils.py based on what argument is passed into the metric parameter of the class.

    Methods:
        fit(X, y)
            Fits the model to the provided data matrix X and targets y.

        predict(X)
            Predicts class target values for the given test data matrix X using the fitted classifier model.
    """

    def __init__(self, n_neighbors = 5, weights = 'uniform', metric = 'l2'):
        # Check if the provided arguments are valid
        if weights not in ['uniform', 'distance'] or metric not in ['l1', 'l2'] or not isinstance(n_neighbors, int):
            raise ValueError('The provided class parameter arguments are not recognized.')

        # Define and setup the attributes for the KNearestNeighbors model object
        self.n_neighbors = n_neighbors
        self.weights = weights
        self._X = None
        self._y = None
        self._distance = euclidean_distance if metric == 'l2' else manhattan_distance
    
    

    def fit(self, X, y):
        """
        Fits the model to the provided data matrix X and targets y.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.
            y: A numpy array of shape (n_samples,) representing the true class values for each sample in the input data.

        Returns:
            None.
        """

        self._X=X
        self._y=y


    def predict(self, X):
        predicted_lables=list()
        for x_test in X:
            distances=[self._distance(x_test,x_train) for x_train in self._X]

            n_indices= np.argsort(distances)[:self.n_neighbors]

            n_nearest_lables=[self._y[i] for i in n_indices]

            if self.weights=='uniform':

                # below line is referenced from https://pythontic.com/containers/counter/most_common
                nearest_neighbor = Counter(n_nearest_lables).most_common(1)        

                predicted_lables.append(nearest_neighbor[0][0])
            
            elif  self.weights=='distance':
                votes=dict()

                for i in range(len(n_nearest_lables)):
                    if votes.get(n_nearest_lables[i]):
                        if distances[n_indices[i]]!=0:
                            votes[n_nearest_lables[i]]+=(1/distances[n_indices[i]])
                        else:
                            votes[n_nearest_lables[i]]+=1
                    else:
                        if distances[n_indices[i]]!=0:
                            votes[n_nearest_lables[i]]=(1/distances[n_indices[i]])
                        else:
                            votes[n_nearest_lables[i]]=1
                
                predicted_lables.append(max(votes,key=votes.get))

        return np.array(predicted_lables)
