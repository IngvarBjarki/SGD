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
    num_in_batch = [1, 2, 5, 10, 50, 75, 100, 150, 200]
    epochs = 1
    k_splits = 5
    learning_rates = [0.0001,  0.001,  0.01, 0.1, 1, 3]
    epsilons = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 10, float('Inf')] # inf makes the noise go to zero -- equal to having no noise
    #epsilon = 200
    weight_decays = [10**(-2), 10**(-5), 10**(-10), 10**(-15)]
    
    parameters = {'learning_rate':[], 'batch_size':[], 'weight_decay':[], 'error_rate':[]}
    optimal_results = {}
    
    
    kf = KFold(n_splits=k_splits)
    for epsilon in epsilons:
        if epsilon not in optimal_results:
            optimal_results[epsilon] = {}
        for n in amount_in_interval:
            # lets do grid search of the parameters for each epsilon
            if n not in optimal_results[epsilon]:
                optimal_results[epsilon][n] = {}
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
                                    noise = utils.add_noise(num_dimensions, epsilon)
                                    
                                
                                    # take a step towrads the optima
                                    weights -= learning_rate *(weight_decay * l2_derivative  + utils.loss_derivative(X_batch, y_batch, weights) / batch_size + noise / batch_size) 
                        
                    
                    
                            # now we predict with the trained weights, using logistic regression
                            num_correct = 0
                            for i in range(len(y_validation)):
                                if y_validation[i] == utils.sigmoid_prediction(X_validation[i], weights):
                                    num_correct += 1
                            avg_error += num_correct/len(y_validation)
            
                        avg_error /= k_splits
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

            
            optimal_results[epsilon][n]['parameters'] = (parameters['learning_rate'][optimal_index]\
                              , parameters['batch_size'][optimal_index], parameters['weight_decay'][optimal_index])
            
            optimal_results[epsilon][n]['error_rate'] = parameters['error_rate'][optimal_index]
            # clear parameters for next run
            parameters = {'learning_rate':[], 'batch_size':[], 'weight_decay':[], 'error_rate':[]}
            
        print('tuning for epsilon: {} done'.format(epsilon))
            
        
            
    return optimal_results
            
#%%
if __name__ == '__main__':
    debugging = False
    small_intervals_in_begining = True
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
    num_samples = len(y)
    if small_intervals_in_begining:
        num_splits = 50
        samples_to_check = 1000
        amount_of_data_in_interval = np.cumsum([int(samples_to_check / num_splits) for i in range(num_splits - 5)]).tolist()
        amount_of_data_in_interval += [2000, 4000, 6000, 8000, num_samples]
    else:
        num_splits = 50
        amount_of_data_in_interval = np.cumsum([int(num_samples / num_splits) for i in range(num_splits)])
    max_integer_val = np.iinfo(np.int32).max
    
    if debugging:
        args = (X, y, amount_of_data_in_interval,  np.random.randint(max_integer_val))
        args2 = (X, y, amount_of_data_in_interval,  np.random.randint(max_integer_val))
        args3 = (X, y, amount_of_data_in_interval,  np.random.randint(max_integer_val))
        all_results = [sgd(args), sgd(args2), sgd(args3)]
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

#%%
    all_results_flatten = {} 
    for result in all_results:
        for epsilon in result:
            if epsilon not in all_results_flatten:
                all_results_flatten[epsilon] = {}
            for n in result[epsilon]:
                if n not in all_results_flatten[epsilon]:
                    all_results_flatten[epsilon][n] = {}
                    all_results_flatten[epsilon][n]['parameters'] = [result[epsilon][n]['parameters']]
                    all_results_flatten[epsilon][n]['error_rate'] = [result[epsilon][n]['error_rate']]
                else:
                    params = result[epsilon][n]['parameters']
                    error_rate = result[epsilon][n]['error_rate']
                    all_results_flatten[epsilon][n]['parameters'].append(params)
                    all_results_flatten[epsilon][n]['error_rate'].append(error_rate)

#%%
    final_results = {}
    for epsilon in all_results_flatten:
        final_results[str(epsilon)] = {}
        for n in all_results_flatten[epsilon]:
            param, error_rate = utils.get_most_common(all_results_flatten[epsilon][n]['parameters'], all_results_flatten[epsilon][n]['error_rate'])
            
            final_results[str(epsilon)][str(n)] = {}
            final_results[str(epsilon)][str(n)]['parameters'] = param
            final_results[str(epsilon)][str(n)]['error_rate'] = error_rate
#%%        
    
    # the json has the key epsilon, numSamples, and value: learning_rate, batch_size, weight_decay
    if small_intervals_in_begining:
        json_string = 'parameters_tight_begining.json'
    else:
        json_string = 'parameters.json'
    with open(json_string, 'w') as f:
        json.dump(final_results, f)
    print('Optimal parameter saved in {}'.format(json_string))
        

    
