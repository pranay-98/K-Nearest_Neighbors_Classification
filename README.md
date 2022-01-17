# K-Nearest_Neighbors_Classification

functions implemented in utils.py:

Eucliedian distance:
The Euclidean distance is the smallest distance between two points in any dimension. It's the square root of the difference between two points' sum of squares.
I have implemeted the Eucliedian distance using np.linalg.norm() function.

Manhattan distance:
The Manhattan distance between two vectors, A and B, is calculated as Σ|A[i] – B[i]|

Implemeting K Nearest Neighbors:

In fit() function assigned the input train data and true class values of train data to the class variables.

In predict() function:

for each sample in the test data find the distance to all the samples in the train data.
Then pick the k_nearest_neighbors and their respective true class values.
And if weight is "Uniform" then each of the k nearest datapoints x_i gets one vote for their corresponding class, and picked the class with the highest votes as the predicted class for the test data point.

if weight is "Distance" then each of the k nearest datapoints x_i is assigned a fraction of a vote for their class inversely proportional to their distance: 1 / d(x_i, x_test), add the inverse distances of same class labels, and pick the class with the highest votes as the predicted class for the test data point. 

and finally returned the all predicted values.
