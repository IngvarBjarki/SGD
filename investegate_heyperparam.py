# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 11:29:42 2018

@author: Ingvar
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns

with open('parameters.json') as f:
    params = json.load(f)

batch_sizes = {}
weight_decays = {}
for epsilon in params:
    batch_sizes[epsilon] = []
    weight_decays[epsilon] = []
    for n in params[epsilon]:
        batch_size = params[epsilon][n]['parameters'][0]
        weight_decay = params[epsilon][n]['parameters'][1]
        
        batch_sizes[epsilon].append((eval(n), batch_size))
        weight_decays[epsilon].append((eval(n), weight_decay))
    
    amount_data, batcher = zip(*sorted(batch_sizes[epsilon]))
    amount_data_b, weight_decaysn = zip(*sorted(weight_decays[epsilon]))
    print('===========================================')
    plt.plot(amount_data, batcher, 'o--')
    plt.xlabel('N')
    plt.ylabel('batch')
    plt.show()
    
    plt.plot(amount_data, weight_decaysn, 'o--', color = 'green')
    plt.xlabel('N')
    plt.ylabel('w-decay')
    plt.show()
    
    plt.plot(batcher, weight_decaysn, 'o', color = 'orange')
    plt.xlabel('batch')
    plt.ylabel('w-decay')
    plt.show()
    
    print('===========================================')