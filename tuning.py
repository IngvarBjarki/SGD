# -*- coding: utf-8 -*-
"""
Created on Sat May 19 17:56:57 2018

@author: Ingvar
"""

import numpy as np
import json
import time
from multiprocessing import Pool
from random import gauss
from sklearn import random_projection
from sklearn.utils import shuffle
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from collections import Counter

# stochastic gradient decent with l2 regularization
# and differential privacy

def get_most_common(parameters, errors):
    # parameters is a list of tuples
    # errors is a list
    # the function returns the parameters which appear most often
    # if there is a tie the one with lower average validation error is returned
    most_common_params = Counter(parameters).most_common(len(parameters))
    most_comon_params = [i for i in most_common_params if most_common_params[0][1] == i[1]]
    if len(most_comon_params) == 1:
        # calc the average error rate
        idx = np.where(np.array(parameters) == most_comon_params[0][0])[0]
        avg_error = np.mean([errors[i] for i in idx])
        return most_comon_params[0][0], avg_error
    else:
        chosen_one = []
        for result in most_common_params:
            idx = np.where(np.array(parameters) == result[0])[0]
            avg_error = np.mean([errors[i] for i in idx])
            chosen_one.append((avg_error, result[0]))
        return min(chosen_one)[1], avg_error # 1 is for acessing the parameters



def get_min_index(error_rate, batch_sizes):
    # error_rate and batch_sizes are both lists
    # get the index with the lowest error rate
    # if there is a tie - select the one with higher batch 
    # as that one is less due to randomization

    error_rate, batch_sizes = np.array(error_rate), np.array(batch_sizes)
    all_min_idx = np.where(error_rate == error_rate.min())[0]
    best_of_ties = np.argmax([batch_sizes[idx] for idx in all_min_idx])
    return all_min_idx[best_of_ties]


def loss_derivative(X, y, w):
    # X, y, w  are np.arrays
    
    # the cross entropy function is the loss function in this case
    # with taregts -1 and 1
    # see http://cseweb.ucsd.edu/~kamalika/pubs/scs13.pdf 
    
    # we use gradient clipping to ensure that the gradient is less or equal than 1
    # we clipp the gradient by the l2 norm
    
    result = 0
    for i in range(len(y)):
        derivative = -(X[i] * y[i] /(np.exp(y[i] * w.T * X[i] ) + 1))
        norm_clipper = max(1, np.linalg.norm(derivative, ord = 2))
        result += derivative / norm_clipper
    return result #sum([-(X[i] * y[i] /(np.exp(y[i] * w.T * X[i] ) + 1)) / max(1, np.linalg.norm((X[i] * y[i] /(np.exp(y[i] * w.T * X[i] ) + 1)), ord = 2)) for i in range(len(y))])

def sigmoid_prediction(X, w):
    # x, y are input arrays
    
    Xw = np.dot(X, w)
    prediction = np.round(1 / (1 + np.exp(-Xw )))
    
    # since the labels are -1 and 1 and sigmoid is in the range 0 to 1
    if prediction == 0.0:
        return -1
    return 1


# taken from: https://stackoverflow.com/questions/6283080/random-unit-vector-in-multi-dimensional-space
def make_rand_vector(dims):
    # dims is an int
    vec = [gauss(0, 1) for i in range(dims)]
    mag = sum(x**2 for x in vec) ** .5
    return [x/mag for x in vec]


def add_noise(learning_rate, batch_size, num_dimensions, epsilon):
    # learning_rate is float, batch_size is int, num_dimensions is int and epsilon is float
    
    # we sample a uniform vector in the unit ball
    v = make_rand_vector(num_dimensions)
    sensitivity = 2
    # sample the magnitude l from gamma (d, senitivity/eps)
    l =  np.array([np.random.gamma(num_dimensions, sensitivity / epsilon) for i in range(num_dimensions)])
    return l * v
    

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
    num_in_batch = [1, 2, 5, 10, 50, 75, 100]
    epochs = 1
    k_splits = 5
    learning_rates = [0.00001, 0.001, 0.01, 0.1, 1, 10]
    epsilons = [1, 0.1, 0.01, 0.0001]
    #epsilon = 200
    weight_decays = [10**(-2), 10**(-5), 10**(-10), 10**(-15)]
    
    parameters = {'learning_rate':[], 'batch_size':[], 'weight_decay':[], 'error_rate':[]}
    all_optimal_results = []
    
    weights = np.array([0.0 for i in range(num_dimensions)])
    
    kf = KFold(n_splits=k_splits)
    for epsilon in epsilons:
        # lets do grid search of the parameters for each epsilon
        optimal_results = {'parameters': [], 'error_rate':[]}
        for n in amount_in_interval:
            for weight_decay in weight_decays:
                for learning_rate in learning_rates:
                    for batch_size in num_in_batch:
                        avg_error = 0
                        for train_index, validation_index in kf.split(X[:n]):
                            X_train, y_train = X[train_index], y[train_index]
                            X_validation, y_validation = X[validation_index], y[validation_index]
                            # shuffle the data so the minibatch takes different data in each epoch
                            X_train, y_train = shuffle(X_train, y_train)
                            for i in range(epochs):
                                for j in range(0, len(y_train), batch_size):
                                    X_batch = X_train[j:j+batch_size]
                                    y_batch = y_train[j:j+batch_size]
                                
                                    # claculate the derative of the l2 norm of the weights 
                                    #l2 = np.linalg.norm(weights, ord = 2)
                                    l2_derivative = sum(weights)
                                    # get the noise for all dimensions
                                    noise = add_noise(learning_rate, batch_size, num_dimensions, epsilon)
                                    
                                
                                    # take a step towrads the optima
                                    weights -= learning_rate *(weight_decay * l2_derivative  + loss_derivative(X_batch, y_batch, weights) / batch_size + noise / batch_size) 
                        
                        #print('epoch: {} out of {}'.format(i, epochs))
                    
                    
                            # now we predict with the trained weights, using logistic regression
                            num_correct = 0
                            for i in range(len(y_validation)):
                                if y_validation[i] == sigmoid_prediction(X_validation[i], weights):
                                    num_correct += 1
                            avg_error += num_correct/len(y_validation)
            
                        avg_error /= k_splits
                        parameters['error_rate'].append(1 - avg_error)
                        parameters['learning_rate'].append(learning_rate)
                        parameters['batch_size'].append(batch_size)
                        parameters['weight_decay'].append(weight_decay)
                        print('epoach..', flush = True)
                        #print('{} out of {} correct with batch size {}, learning_rate: {}'.format(num_correct, len(y_validation), batch_size, learning_rate))
            #print('=========================================================================')
            #print('error rate', parameters['error_rate'])
            #print('batch_size', parameters['batch_size'])        
            #print('=========================================================================')
            # find the optimal parameters fro the cross validation --
            optimal_index = get_min_index(parameters['error_rate'], parameters['batch_size'])
            
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
        
    else:
        num1, num2 = 4, 9
        y_train, X_train = [], []
        with open('mnist_train.csv') as l:
            for i, line in enumerate(l):
                line = line.split(',')
                label = int(line[0])
                if label == num1 or label == num2:
                    features = [float(i) for i in line[1:]]
                    y_train.append(label)
                    X_train.append(features)
        y_train = np.asarray(y_train)
        X_train = np.asarray(X_train)
        
        y_train[y_train == num1] = -1
        y_train[y_train == num2] = 1
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
    
    
    # split the data upp so to get the learning rate
    num_splits = 30
    num_samples = len(y)
    amount_of_data_in_interval = np.cumsum([int(num_samples / num_splits) for i in range(num_splits)])
    
    max_integer_val = np.iinfo(np.int32).max
    
    num_processes = 24
    args = [(X, y, amount_of_data_in_interval,  np.random.randint(max_integer_val)) for i in range(num_processes)] 
    t1 = time.time()
    p = Pool(num_processes)
    
    all_results = p.map(sgd, args)# [sgd(X, y, amount_of_data_in_interval,  np.random.randint(max_integer_val))] 
    #%%
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
        param, error_rate = get_most_common(all_results_flatten[key]['parameters'], all_results_flatten[key]['error_rate'])
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
    
    

    
    
print(get_most_common(['ingvar', 'ingvar', 'sigurd', 'bard', 'bard'], [1,2,3,4,5]))
    
