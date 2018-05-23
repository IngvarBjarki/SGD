# -*- coding: utf-8 -*-
"""
Created on Tue May 22 12:19:58 2018

@author: helga
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May 19 17:56:57 2018

@author: Ingvar
"""

import numpy as np
import time
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn import random_projection
from sklearn.preprocessing import StandardScaler
from random import gauss
# stochastic gradient decent with l2 regularization
# and differential privacy









def loss_derivative(X, y, w):
    # the cross entropy function is the loss function in this case
    # with taregts -1 and 1
    # see http://cseweb.ucsd.edu/~kamalika/pubs/scs13.pdf 
    
    # we need to bound it ... gradient clipping? norm ma ekki vera stlrra en 1..
    #print([-(X[i] * y[i] /(np.exp(y[i] * w.T * X[i] ) + 1))  for i in range(len(y))])
    # note we clip the gradient by the l2 norm if it is greater than 1
    result = 0
    for i in range(len(y)):
        derivative = -(X[i] * y[i] /(np.exp(y[i] * w.T * X[i] ) + 1))
        norm_clipper = max(1, np.linalg.norm(derivative, ord = 2))
        result += derivative / norm_clipper
    return result #sum([-(X[i] * y[i] /(np.exp(y[i] * w.T * X[i] ) + 1)) / max(1, np.linalg.norm((X[i] * y[i] /(np.exp(y[i] * w.T * X[i] ) + 1)), ord = 2)) for i in range(len(y))])

def sigmoid_prediction(X, w):
    Xw = np.dot(X, w)
    prediction = np.round(1 / (1 + np.exp(-Xw )))
    
    # since the labels are -1 and 1 and sigmoid is in the range 0 to 1
    if prediction == 0.0:
        return -1
    return 1


# taken from: https://stackoverflow.com/questions/6283080/random-unit-vector-in-multi-dimensional-space
def make_rand_vector(dims):
    vec = [gauss(0, 1) for i in range(dims)]
    mag = sum(x**2 for x in vec) ** .5
    return [x/mag for x in vec]


def gamma_noise(learning_rate, batch_size, num_dimensions, epsilon):
    # we sample a uniform vector in the unit ball
    v = make_rand_vector(num_dimensions)
    sensitivity = 2
    # sample the magnitude l from gamma (d, senitivity/eps)
    l =  np.array([np.random.gamma(num_dimensions, sensitivity / epsilon) for i in range(num_dimensions)])
    #print('l',l)
    #print('v', v)
    return l * v
    
    # same algrithm is used in 13..
    return 0

def sgd(X, y):
    # X are the predictors, come as np array
    # y are the targets, come as np array
    
    # split train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    
    num_dimensions = len(X[0])
    num_in_batch = [1, 2, 5, 10, 50]
    epochs = 1
    learning_rates = [0.01]
    epsilons = [10]
    #epsilon = 200
    weight_decay = 10**(-7)
    
    optimal_parameters = {'learning_rate':[], 'batch_size':[], 'weight_decay':[]}
    
    weights = np.array([0.0 for i in range(num_dimensions)])
    # optimize the weights
    for epsilon in epsilons:
        for learning_rate in learning_rates:
            for batch_size in num_in_batch:
                # shuffle the data so the minibatch takes different data in each epoch
                X_train, y_train = shuffle(X_train, y_train)
                for i in range(epochs):
                    for j in range(0, len(y_train), batch_size):
                        X_batch = X_train[j:j+batch_size]
                        y_batch = y_train[j:j+batch_size]
                        
                        # claculate the l2 norm of the weights
                        l2 = np.linalg.norm(weights, ord = 2)
                        
                        # get the noise for all dimensions
                        noise = gamma_noise(learning_rate, batch_size, num_dimensions, epsilon)
                        #print('noise', sum(noise))
                        
                        # take a step towrads the optima
                        weights -= learning_rate *(weight_decay * l2 + loss_derivative(X_batch, y_batch, weights) / batch_size + noise / batch_size) 
                
            #print('epoch: {} out of {}'.format(i, epochs))
            
            
                    # now we predict with the trained weights, using logistic regression
                    num_correct = 0
                    for i in range(len(y_test)):
                        if y_test[i] == sigmoid_prediction(X_test[i], weights):
                            num_correct += 1
                            
                    print('{} out of {} correct with batch size {}'.format(num_correct, len(y_test), batch_size))
                    
    return results
            
#%%
if __name__ == '__main__':
    
    # get the data and preprocess it
    digits = load_digits()
    n_samples = len(digits.images)
    X_without_bias = digits.images.reshape((n_samples, -1))
    y = digits.target
     
    # now we only want to do binary classification of two numbers
    # so we take only number 0 and 2 ---- 9 and 4 are probably most similar
    index_of_zeros =  np.flatnonzero( y == 4 ) #returns the indexes
    index_of_tows = np.flatnonzero( y == 9 )

    # merge the two together and  sort them
    new_indexes = np.concatenate((index_of_zeros, index_of_tows), axis=0)
    new_indexes = np.sort(new_indexes)
    y = y[new_indexes]
    X_without_bias = X_without_bias[new_indexes]
    # since we are classifying with the sign - we translate the y vector  to -1 to 1
    y[y == 4] = -1
    y[y == 9] = 1
    
    # standardize the data
    scaler = StandardScaler()
    scaler.fit(X_without_bias)
    X_without_bias = scaler.transform(X_without_bias)
    
    # do the random projection as they do in the paper -- second paper
    transformer = random_projection.GaussianRandomProjection(n_components = 50)
    X_without_bias = transformer.fit_transform(X_without_bias)
    
    # we add bias term in front -- done for the gradient decent
    records, attributes = np.shape(X_without_bias)
    X = np.ones((records, attributes + 1))
    X[:,1:] = X_without_bias
    
    
    sgd(X,y)
    print('done')
    
    
    
