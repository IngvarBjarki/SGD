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

with open('results.json') as f:
    results = json.load(f)


with open('objective_info.json') as f:
    objective_info = json.load(f)
    
    
num_simulations = 2
t_critical = stats.t.ppf(q = 0.95, df = num_simulations - 1)


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
                'inf': 'OPtimal..'
        }


#%%
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
    plt.errorbar(num_points, error_rates_mean, yerr = limits, label =get_label[epsilon] ,  fmt = '--o', capsize = 2, markersize = 5)   
    #plt.plot(num_points, error_rates_mean, 'o--')
plt.legend(bbox_to_anchor=(1.05, 0.85), loc=2, borderaxespad=0.)
box = ax.get_postion()
ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
plt.ylabel('Error rate')
plt.xlabel('Amount of training data [N]')
plt.show()

#%%
# look at the size of the error bar



# look at the noise magnitude
for epsilon in results:
    
    for i, n in enumerate(results[epsilon]):
        
        if i == 0 or i == int(len(results[epsilon]) / 2) or i == int(len(results[epsilon]) ):
            noise_magnitude = [np.mean(results[epsilon][n]['noise_magnitude'])]
            plt.plot([int(n)], noise_magnitude)


plt.show()

# look at the noise distribution




# look at the convergance of the objective function
for epsilon in objective_info:
    # hversu margir punktar gerdu tetta...
    
    
# look at the gradient
for epsilon in objective_info:
    gradient = ...
