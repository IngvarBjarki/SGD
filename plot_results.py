# -*- coding: utf-8 -*-
"""
Created on Mon May 28 10:05:15 2018

@author: Ingvar
"""
#%%
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as stats

colors = sns.color_palette('Set1', n_colors = 9) + [(1.0, 191/255, 0.0)] + sns.color_palette('Set2', n_colors = 3)[0:3:2]
sns.set_palette(colors)
sns.set_style('darkgrid')
# get the data

with open(r"C:\Users\helga\OneDrive\Documents\sgd\results.json") as f:
    results = json.load(f)


with open(r"C:\Users\helga\OneDrive\Documents\sgd\objective_info.json") as f:
    objective_info = json.load(f)
    
    
num_simulations = 4
t_critical = stats.t.ppf(q = 0.95, df = num_simulations - 1)

#!! s;mrama optimal!!!!!!!
get_label = {
                '0.0005':'$\epsilon$ = 0.0005',
                '0.001': '$\epsilon$ = 0.001',
                '0.005' : '$\epsilon$ = 0.005',
                '0.01': '$\epsilon$ = 0.01',
                '0.05': '$\epsilon$ = 0.05',
                '0.1': '$\epsilon$ = 0.1',
                '0.5': '$\epsilon$ = 0.5',
                '1': '$\epsilon$ = 1',
                '10': '$\epsilon$ = 10',
                'inf': 'Without DP',
                'Infinity': 'Without DP'
        }


#%%
for i in range(2):
    if i == 1:
        print('plot on log axis')
    all_limits = []
    fig = plt.figure(figsize=(7, 4))
    ax = plt.subplot(111)
    # look at the error rate    
    for epsilon in results:
        error_rates_mean = []
        error_rate_interval = []
        num_points = []
        limits = []
        for n in results[epsilon]:
            num_points.append(int(n))
            error_rate_mean = np.mean(results[epsilon][n]['error_rate'])
            
            # find confidance interval
            error_rate_mean_std = np.std(results[epsilon][n]['error_rate'])
            limit = t_critical * error_rate_mean_std / np.sqrt(num_simulations)
            
            error_rates_mean.append(error_rate_mean)
            limits.append(limit)
        all_limits.append((epsilon, list(limits)))
        #plt.errorbar(num_points, error_rates_mean, yerr = limits, label =get_label[epsilon] ,  fmt = '--o', capsize = 2, markersize = 5)   
        plt.plot(num_points, error_rates_mean, 'o--', label =get_label[epsilon])
    plt.legend(bbox_to_anchor=(1.05, 0.85), loc=2, borderaxespad=0.)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    plt.xlabel('Amount of training data [N]')
    
    if i == 1:
        plt.yscale('log')
        plt.ylabel('log(Error rate)')
    else:
        plt.ylabel('Error rate')
    
    plt.show()
    
    # look at the size of the error bar
    fig = plt.figure(figsize=(7, 4))
    ax = plt.subplot(111)
    for epsilon, limit in all_limits:
        plt.plot(limit, '*--', label=get_label[epsilon])
    plt.legend(bbox_to_anchor=(1.05, 0.85), loc=2, borderaxespad=0.)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    
    plt.xlabel('Amount of training data [N]')
    if i == 1:
        plt.yscale('log')
        plt.ylabel('log(upper limit - lower limit)') #!!!!!!!!!!! samr;mma!!
    else:
        plt.ylabel('upper limit - lower limit') #!!!!!!!!!!! samr;mma!!
    plt.show()
    

#%%

plotting_n = [0, int(len(results[epsilon]) / 2), int(len(results[epsilon])) - 1]
for i in range(len(plotting_n)):
    noise_magnitude = []
    x_labels = []
    # look at the noise magnitude
    for epsilon in results:
        
        for j, n in enumerate(results[epsilon]):
            
            if j == plotting_n[i]:
                if epsilon == '1':
                    # print once...
                    print('\n n = {}'.format(n))
                
                
                #noise_magnitude.append(results[epsilon][n]['weights'])
                noise_magnitude.append(results[epsilon][n]['noise_and_weights_magnitude'])
    
        x_labels.append(get_label[epsilon])
    
    
    
    sns.barplot(data = noise_magnitude, estimator = sum)
    plt.yscale('log')
    plt.xticks(range(len(x_labels)), x_labels, rotation=45)
    plt.show()





#%%
# look at how the noise influences the weights by injecting into the training
plotting_n = [0, int(len(results[epsilon]) / 2), int(len(results[epsilon])) - 1]
for i in range(len(plotting_n)):
    noise_magnitude = []
    x_labels = []
    # look at the noise magnitude
    for epsilon in results:
        
        for j, n in enumerate(results[epsilon]):
            
            if j == plotting_n[i]:
                if epsilon == '1':
                    # print once...
                    print('\n n = {}'.format(n))
                
                
                noise_magnitude.append(results[epsilon][n]['weights'])
                #noise_magnitude.append(results[epsilon][n]['noise_and_weights_magnitude'])
    
        x_labels.append(get_label[epsilon])
    
    
    
    sns.barplot(data = noise_magnitude, estimator = sum)
    plt.yscale('log')
    plt.xticks(range(len(x_labels)), x_labels, rotation=45)
    plt.show()




#%%
# =============================================================================
# # look at the noise distribution
# plotting_n = [0, int(len(results[epsilon]) / 2), int(len(results[epsilon])) - 1]
# for i in range(len(plotting_n)):
#     noise = [[] for i in range(len(results))]
#     x_labels = []
#     # look at the noise magnitude
#     for j, epsilon in enumerate(results):
#         
#         for k, n in enumerate(results[epsilon]):
#             
#             if k == plotting_n[i]:
#                 if epsilon == '1':
#                     # print once...
#                     print('\n n = {}'.format(n))
#                 noise[j] += [np.arcsinh(i) for i in results[epsilon][n]['noise']]
#     
#         x_labels.append(get_label[epsilon])
#     
#     
#     
#     sns.boxplot(data = noise)
#     plt.xticks(range(len(x_labels)), x_labels, rotation=45)
#     plt.show()
# =============================================================================



#%%
# look at the convergance of the objective function
for i in range(2):
    all_limits = []
    if i == 1:
        print('plot on log axis')
    fig = plt.figure(figsize=(7, 4))
    ax = plt.subplot(111)
    for epsilon in objective_info:
        # hversu margir punktar gerdu tetta... num sim..
        means = [np.mean(objective_info[epsilon]['objective'][i]) for i in range(len(objective_info[epsilon]['objective']))]
        stds = [np.std(objective_info[epsilon]['objective'][i]) for i in range(len(objective_info[epsilon]['objective']))]
        limits = [t_critical * std / np.sqrt(num_simulations - 1) for std in stds]
        all_limits.append((epsilon,limits))
        num_points = objective_info[epsilon]['num_points']
        plt.plot(num_points, means, label = get_label[epsilon])
        #plt.errorbar(num_points, means, yerr = limits, label = get_label[epsilon] ,  fmt = '--o', capsize = 2, markersize = 5)
    plt.legend(bbox_to_anchor=(1.05, 0.85), loc=2, borderaxespad=0.)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    plt.xlabel('Amount of training data [N]')
    if i == 1:
        plt.yscale('log')
        plt.ylabel('log(Objective value)')
    else:
        plt.ylabel('Objective value')
    plt.show()
    
    print('look at CI')
    # look at the size of the error bar
    fig = plt.figure(figsize=(7, 4))
    ax = plt.subplot(111)
    for epsilon, limit in all_limits:
        num_points = objective_info[epsilon]['num_points']
        plt.plot(num_points, limit, '*--', label=get_label[epsilon])
    plt.legend(bbox_to_anchor=(1.05, 0.85), loc=2, borderaxespad=0.)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    
    plt.xlabel('Amount of training data [N]')
    if i == 1:
        plt.yscale('log')
        plt.ylabel('log(upper limit - lower limit)') #!!!!!!!!!!! samr;mma!!
    else:
        plt.ylabel('upper limit - lower limit') #!!!!!!!!!!! samr;mma!!
    plt.show()

#%%   
# look at the gradient
for i in range(2):
    all_limits = []
    if i == 1:
        print('plot on log axis')
    fig = plt.figure(figsize=(7, 4))
    ax = plt.subplot(111)
    for epsilon in objective_info:
        # hversu margir punktar gerdu tetta... num sim..
        means = [np.mean(objective_info[epsilon]['gradient'][i]) for i in range(len(objective_info[epsilon]['gradient']))]
        stds = [np.std(objective_info[epsilon]['gradient'][i]) for i in range(len(objective_info[epsilon]['gradient']))]
        limits = [t_critical * std / np.sqrt(num_simulations - 1) for std in stds]
        all_limits.append((epsilon, limits))
        num_points = objective_info[epsilon]['num_points']
        
        
        plt.plot(num_points, means, '<--',label = get_label[epsilon])
        #plt.errorbar(num_points, means, yerr = limits, label = get_label[epsilon] ,  fmt = '--o', capsize = 2, markersize = 5)
    plt.legend(bbox_to_anchor=(1.05, 0.85), loc=2, borderaxespad=0.)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    plt.xlabel('Amount of training data [N]')
    if i == 1:
        plt.yscale('log')
        plt.ylabel('log(Gradient)')
    else:
        plt.ylabel('gradient')
    plt.show()
    
    print('look at CI')
    # look at the size of the error bar
    fig = plt.figure(figsize=(7, 4))
    ax = plt.subplot(111)
    for epsilon, limit in all_limits:
        num_points = objective_info[epsilon]['num_points']
        plt.plot(num_points, limit, '*--', label=get_label[epsilon])
    plt.legend(bbox_to_anchor=(1.05, 0.85), loc=2, borderaxespad=0.)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    
    plt.xlabel('Amount of training data [N]')
    if i == 1:
        plt.yscale('log')
        plt.ylabel('log(upper limit - lower limit)') #!!!!!!!!!!! samr;mma!!
    else:
        plt.ylabel('upper limit - lower limit') #!!!!!!!!!!! samr;mma!!
    plt.show()
    
#%%
# COUNT how many times each attribute comes upp after grid search