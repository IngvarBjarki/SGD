# -*- coding: utf-8 -*-
"""
Created on Sat May 19 17:56:57 2018

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
from sklearn.model_selection import KFold

# my libraries
import utils

# stochastic gradient decent with l2 regularization
# and differential privacy


    

def sgd(all_input_params):
    X, y, amount_in_interval, random_state = all_input_params
    # X are the predictors, come as np array
    # y are the targets, come as np array
    # amount_in_interval is the number of samples used to geneerate learning curve
    
   
    
    # multiprocessing dose not do different seed, so we take a random number to start different seeds
    np.random.seed(random_state)
    
    # shuffle so different data will be used in each process
    X, y = shuffle(X, y)
    
    num_dimensions = len(X[0])
    num_in_batch = [1, 2, 5, 10, 50, 75, 100, 150]
    epochs = 1
    k_splits = 5
    learning_rates = [0.0001, 0.005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 3]
    epsilons = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 10, float('Inf')] # inf makes the noise go to zero -- equal to having no noise
    #epsilon = 200
    weight_decays = [10**(-2), 10**(-5), 10**(-10), 10**(-15)]
    
    parameters = {'learning_rate':[], 'batch_size':[], 'weight_decay':[], 'error_rate':[]}
    all_optimal_results = []
    
    
    
    kf = KFold(n_splits=k_splits)
    for epsilon in epsilons:
        for n in amount_in_interval:
            # lets do grid search of the parameters for each epsilon
            optimal_results = {'parameters': [], 'error_rate':[]}
            for weight_decay in weight_decays:
                for learning_rate in learning_rates:
                    for batch_size in num_in_batch:
                        avg_error = 0
                        for train_index, validation_index in kf.split(X[:n]):
                            X_train, y_train = X[train_index], y[train_index]
                            X_validation, y_validation = X[validation_index], y[validation_index]
                            weights = np.array([0.0 for i in range(num_dimensions)])
                            for i in range(epochs):
                                # shuffle the data so the minibatch takes different data in each epoch
                                X_train, y_train = shuffle(X_train, y_train)
                                for j in range(0, len(y_train), batch_size):
                                    X_batch = X_train[j:j+batch_size]
                                    y_batch = y_train[j:j+batch_size]
                                
                                    # claculate the derative of the l2 norm of the weights 
                                    #l2 = np.linalg.norm(weights, ord = 2)
                                    l2_derivative = sum(weights)
                                    # get the noise for all dimensions
                                    noise = utils.add_noise(learning_rate, batch_size, num_dimensions, epsilon)
                                    
                                
                                    # take a step towrads the optima
                                    weights -= learning_rate *(weight_decay * l2_derivative  + utils.loss_derivative(X_batch, y_batch, weights) / batch_size + noise / batch_size) 
                        
                        #print('epoch: {} out of {}'.format(i, epochs))
                    
                    
                            # now we predict with the trained weights, using logistic regression
                            num_correct = 0
                            for i in range(len(y_validation)):
                                if y_validation[i] == utils.sigmoid_prediction(X_validation[i], weights):
                                    num_correct += 1
                            avg_error += num_correct/len(y_validation)
            
                        avg_error /= k_splits
                        print('1 - avg_error {}, lr {} bs {} wd {}'.format(1 - avg_error, learning_rate, batch_size, weight_decay))
                        parameters['error_rate'].append(1 - avg_error)
                        parameters['learning_rate'].append(learning_rate)
                        parameters['batch_size'].append(batch_size)
                        parameters['weight_decay'].append(weight_decay)
                        #print('epoach..', flush = True)
                        #print('{} out of {} correct with batch size {}, learning_rate: {}'.format(num_correct, len(y_validation), batch_size, learning_rate))
            #print('=========================================================================')
            #print('error rate', parameters['error_rate'])
            #print('batch_size', parameters['batch_size'])        
            #print('=========================================================================')
            
            # find the optimal parameters fro the cross validation --
            optimal_index = utils.get_min_index(parameters['error_rate'], parameters['batch_size'])
            print('epsilon {}, n {}'.format(epsilon, n))
            print(parameters['error_rate'])
            print('len(parameters[error_rate])', len(parameters['error_rate']))
            
            optimal_results['parameters'].append((parameters['learning_rate'][optimal_index]\
                              , parameters['batch_size'][optimal_index], parameters['weight_decay'][optimal_index]))
            optimal_results['error_rate'].append(parameters['error_rate'][optimal_index])
            # clear parameters for next run
            parameters = {'learning_rate':[], 'batch_size':[], 'weight_decay':[], 'error_rate':[]}
            # save the optimal parameters for each epsilon
        all_optimal_results.append((epsilon, optimal_results))
            
    return all_optimal_results
            
#%%
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
    num_splits = 1
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
    # nafnid a splitinu er key
    all_results_flatten = {} 
    for result in all_results:
        for i, values in enumerate(result):
            epsilon = values[0]
            parameters = values[1]['parameters']
            print('parameters ', parameters )
            error_rate = values[1]['error_rate']
            for i in range(len(parameters)):
                key = (epsilon, amount_of_data_in_interval[i])
                if key not in all_results_flatten:
                    all_results_flatten[key] = {}
                    all_results_flatten[key]['parameters'] = [parameters[i]]
                    all_results_flatten[key]['error_rate'] = [error_rate[i]]
                else:
                    all_results_flatten[key]['parameters'].append(parameters[i])
                    all_results_flatten[key]['error_rate'].append(error_rate[i])
            
            print('***********************************')
#%%            
    # find which parameter is most common
    final_results = {}
    for key in all_results_flatten:
        param, error_rate = utils.get_most_common(all_results_flatten[key]['parameters'], all_results_flatten[key]['error_rate'])
        key = str(key)
        final_results[key] = {}
        final_results[key]['parameters'] = param
        final_results[key]['error_rate'] = error_rate
        
    
    # the json has the key epsilon, numSamples, and value: learning_rate, batch_size, weight_decay
    json_string = 'parameters.json'
    with open(json_string, 'w') as f:
        json.dump(final_results, f)
    print('Optimal parameter saved in {}'.format(json_string))
        
    # maybe we would like to get the batcheses which are most often used.
    
    
#%%
    
    
