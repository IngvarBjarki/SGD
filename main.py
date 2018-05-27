# -*- coding: utf-8 -*-
"""
Created on Sat May 26 18:30:51 2018

@author: Ingvar
"""
import numpy as np
import json
import time
from multiprocessing import Pool
from sklearn import random_projection
from sklearn.utils import shuffle
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

# my libraries
import utils





if __name__ == '__main__':
    debugging = False
    if debugging:
        # get the data and preprocess it
        digits = load_digits()
        n_samples = len(digits.images)
        X_without_bias = digits.images.reshape((n_samples, -1))
        y = digits.target
         
        # now we only want to do binary classification of two numbers
        num1, num2 = 4, 9
        
        index_of_num1 =  np.flatnonzero( y == num1 ) #returns the indexes
        index_of_num2 = np.flatnonzero( y == num2 )
    
        # merge the two together and  sort them
        new_indexes = np.concatenate((index_of_num1, index_of_num2), axis=0)
        new_indexes = np.sort(new_indexes)
        y = y[new_indexes]
        X_without_bias = X_without_bias[new_indexes]
        # since we are classifying with the sign - we translate the y vector  to -1 to 1
        y[y == num1] = -1
        y[y == num2] = 1
        
    else:
        num1, num2 = 4, 9
        y, X_without_bias = [], []
        with open('mnist_train.csv') as l:
            for i, line in enumerate(l):
                line = line.split(',')
                label = int(line[0])
                if label == num1 or label == num2:
                    features = [float(i) for i in line[1:]]
                    y.append(label)
                    X_without_bias.append(features)
        y = np.asarray(y)
        X_without_bias = np.asarray(X_without_bias)
        
        y[y == num1] = -1
        y[y == num2] = 1
    # standardize the data
    scaler = StandardScaler()
    scaler.fit(X_without_bias)
    X_without_bias = scaler.transform(X_without_bias)
    
    # project the data onto the unitball
    X_without_bias = utils.project_onto_unitball(X_without_bias)
    
    
    # do the random projection as they do in the paper -- second paper
    transformer = random_projection.GaussianRandomProjection(n_components = 50)
    X_without_bias = transformer.fit_transform(X_without_bias)
    
    # we add bias term in front -- done for the gradient decent
    records, attributes = np.shape(X_without_bias)
    X = np.ones((records, attributes + 1))
    X[:,1:] = X_without_bias
    
    
    # split the data upp so to get the learning rate
    num_splits = 50
    num_samples = len(y)
    amount_of_data_in_interval = np.cumsum([int(num_samples / num_splits) for i in range(num_splits)])
    max_integer_val = np.iinfo(np.int32).max
    
    if debugging:
        args = (X, y, amount_of_data_in_interval,  np.random.randint(max_integer_val))
        all_results = [sgd(args)]
    else:
        # we run mulitiprocessing when we are not debuging
        num_processes = 24
        args = [(X, y, amount_of_data_in_interval,  np.random.randint(max_integer_val)) for i in range(num_processes)] 
        t1 = time.time()
        p = Pool(num_processes)
    
        all_results = p.map(sgd, args)# [sgd(X, y, amount_of_data_in_interval,  np.random.randint(max_integer_val))] 
    
        p.close()
        p.join()
        print('multiporcessing finsihed, time: {}'.format(time.time() - t1))
