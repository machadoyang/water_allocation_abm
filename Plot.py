# -*- coding: utf-8 -*-
"""
@author: machadoyang
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import numpy as np

plt.style.use('seaborn-white')

def total_revenue(agents_results):
    farmers_results = agents_results[agents_results['Type'] == 'farmer']
    fig, ax = plt.subplots(dpi=300)
    index = []
    value = []
    water_value = []
    for i in range(farmers_results.index.get_level_values(0).min(), farmers_results.index.get_level_values(0).max()):
        print(i)
        results_current_step = farmers_results.xs(i, level=0)
        index.append(i)
        value.append(results_current_step['Total revenue (R$)'].sum()/1000000)
        water_value.append(results_current_step['Amount of water withdrawn (m³/year)'].sum())
    ax.plot(index, value, color = 'black')
    ax.set_ylim(0,600)
    ax.set_xlabel('Step')
    ax.set_ylabel('Total revenue (million R$)')
    ax2 = ax.twinx() # Secondary y axis
    ax2.set_ylim(0,40000000)
    ax2.set_ylabel('Water withdrawn (m³/year)')
    plt.rcParams.update({'font.size': 16})
    print(value)
    print(water_value)
    ax2.plot(index,
            water_value,
            color='blue')
    

def agents_impact_over_years(agents_results):
    farmers_results = agents_results[agents_results['Type'] == 'farmer']
    fig, ax = plt.subplots(dpi=300)
    # for i in range(farmers_results.index.get_level_values(1).min(), farmers_results.index.get_level_values(1).max()):
    results_current_agent = farmers_results.xs(4, level=1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('step')
    ax.set_ylabel('Total revenue (R$)')
    # ax.set_yscale('log')
    ax.set_ylim(0,20000)
    ax.plot(results_current_agent['Total revenue (R$)'].index,
             results_current_agent['Total revenue (R$)'].values, color='black', alpha=0.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    print(results_current_agent)
    # for i in range(1,4):
    #     results_current_agent = farmers_results.xs(i, level=1)
    #     ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    #     ax.set_xlabel('step')
    #     ax.set_ylabel('Total revenue (R$)')
    #     ax.set_yscale('log')
    #     ax.plot(results_current_agent['Total profit (R$)'].index,
    #              results_current_agent['Total profit (R$)'].values, color='black', alpha=0.5)
    #     ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    #     print(results_current_agent)
        
def override_plot(agents_results):
    farmers_results = agents_results[agents_results['Type'] == 'farmer']
    farmers_who_override = farmers_results.loc[farmers_results['Amount of water withdrawn (m³/year)'] > farmers_results['Amount of water received (m³/year)']]
    farmers_had_water_right_but_no_water_withdrawn = farmers_results.loc[farmers_results['Amount of water received (m³/year)'] > farmers_results['Amount of water withdrawn (m³/year)']]
    
    farmers_who_override_no_duplicates = farmers_who_override.drop_duplicates(subset='Position', keep="first")
    farmers_had_water_right_but_no_water_withdrawn_no_duplicates = farmers_had_water_right_but_no_water_withdrawn.drop_duplicates(subset='Position', keep="first")
    
    fig, ax = plt.subplots(dpi=300)
    ax.set_yscale('log')
    ax.set_xlabel('step')
    ax.set_ylabel('Water volume (m³/year)')
    ax2 = ax.twinx() # Secondary y axis
    # ax2.set_yscale('log')
    ax2.set_ylabel('N. of agents')
    
    # Count number of agents overriden each step
    n_agents_overriden = farmers_who_override.index.get_level_values('Step').value_counts().sort_index(ascending=True)
    
    # Count number of agents no water even thought water right was conceived
    n_agents_no_water = farmers_had_water_right_but_no_water_withdrawn.index.get_level_values('Step').value_counts().sort_index(ascending=True)
    
    # Plot override
    ax.set_ylim(0,1000000)
    ax.set_xlim(1,20)
    ax.scatter(farmers_who_override_no_duplicates.index.get_level_values('Step'),
              farmers_who_override_no_duplicates['Amount of water withdrawn (m³/year)'].values,
              color='red', alpha=0.3, label="Farmer override")
    
    # Plot no water even thought water right was conceived
    ax.scatter(farmers_had_water_right_but_no_water_withdrawn_no_duplicates.index.get_level_values('Step'),
             farmers_had_water_right_but_no_water_withdrawn_no_duplicates['Amount of water asked (m³/year)'].values,
             facecolors='none', edgecolors='black', alpha=0.3, label="Farmer deceived")
    
    print(n_agents_overriden)
    print(n_agents_no_water)
    plt.rcParams.update({'font.size': 18})
    # Plot count number of agents overriden
    ax2.plot(n_agents_overriden.index,
            n_agents_overriden.values,
            color='red', alpha=0.6, label="Farmer override")
    
    # Plot count number of agents no water
    ax2.set_ylim(0,300)
    ax2.plot(n_agents_no_water.index,
            n_agents_no_water.values,
            color='black', alpha=0.6, label="Farmer deceived")   
    
    plt.legend(loc="lower left")
    return farmers_who_override.index.get_level_values('Step').value_counts().sort_index(ascending=True)

def water_level_in_canal(model_results, init_water):
    result = model_results[0]
    # Calculate total water volume available
    total_water_available = sum(init_water.values())
    
    # Fill all zeros with previous section water availability
    result['water_available'] = result['water_available'].replace(to_replace=0, method='ffill')
    
    # Convert Step column into int and apply multiindex to dataFrame
    result['section'] = result['section'].astype(int)
    result_multiindex = result.set_index(['step', 'section'])
    
    # Prepare plotting
    number_of_subplots = int(result_multiindex.index.get_level_values(1).max())
    cols = 3
    rows = number_of_subplots // cols 
    rows += number_of_subplots % cols
    
    position = range(1, number_of_subplots + 1)
    
    fig = plt.figure(1, figsize=(16, 18))
    plt.rcParams.update({'font.size': 14})
      
    # Iterate through sections and plot
    for k in range(number_of_subplots):
        results_current_step = result_multiindex.xs(k+1, level=1)/total_water_available*100
        # add every single subplot to the figure with a for loop
        ax = fig.add_subplot(rows, cols , position[k])
        ax.set_ylim(0,100)
        if (k in list(range(number_of_subplots)[-3:])):
            ax.set(xlabel='Step')
        if (k==0):
            ax.set(ylabel='Water volume (%)')
        else:
            if ((k % cols) == 0 and k != 1):
                ax.set(ylabel='Water volume (%)')
        ax.plot(results_current_step.index, results_current_step['water_available'].values, color='blue')
    return result_multiindex
        
        
def virtual_water_level_in_canal(model_results, init_water):
    
    # Convert Step column into int and apply multiindex to dataFrame
    model_results['Section'] = model_results['Section'].astype(int)
    model_results_multiindex = model_results.set_index(['Step', 'Section'])
    
    # Prepare plotting
    number_of_subplots = int(model_results_multiindex.index.get_level_values(1).max())
    cols = 3
    rows = number_of_subplots // cols 
    rows += number_of_subplots % cols
    
    position = range(1, number_of_subplots + 1)
    
    fig = plt.figure(1, figsize=(16, 18))
    plt.rcParams.update({'font.size': 14})
      
    # Iterate through sections and plot
    for k in range(number_of_subplots):
        results_current_step = model_results_multiindex.xs(k+1, level=1)/init_water[str(k+1)]*100
        # print(k)
        # add every single subplot to the figure with a for loop
        ax = fig.add_subplot(rows, cols , position[k])
        ax.set_ylim(0,100)
        if (k in list(range(number_of_subplots)[-3:])):
            ax.set(xlabel='Step')
        if (k==0):
            ax.set(ylabel='Virtual water (%)')
        else:
            if ((k % cols) == 0 and k != 1):
                print(k)
                ax.set(ylabel='Virtual water (%)')
        ax.plot(results_current_step.index, results_current_step['Virtual Water Available'].values, color='black')
        
        
def real_and_virtual_water_same_axis(model_results, init_water):
    """ Real water"""
    # Calculate total water volume available
    total_water_available = sum(init_water.values())
    
    model_results_multiindex = []
    for result in model_results:
        # Fill all zeros with previous section water availability
        result['water_available'] = result['water_available'].replace(to_replace=0, method='ffill')
        
        # Convert Step column into int and apply multiindex to dataFrame
        result['section'] = result['section'].astype(int)
        model_results_multiindex.append(result.set_index(['step', 'section']))
    
    # Prepare plotting
    number_of_subplots = int(model_results_multiindex[0].index.get_level_values(1).max())
    cols = 3
    rows = number_of_subplots // cols 
    rows += number_of_subplots % cols
    position = range(1, number_of_subplots + 1)
    
    fig, ax = plt.subplots(rows, cols, figsize=(16, 18))
    plt.rcParams.update({'font.size': 14})
    ax = ax.flatten()
    
    for k in range(number_of_subplots):
        ax[k].set_ylim(0, 100)
        if k >= number_of_subplots - cols:
            ax[k].set(xlabel='Step')
        if k % cols == 0:
            ax[k].set(ylabel='Water volume (%)')
    
    for result_multiindex in model_results_multiindex:
        #Iterate through sections and plot
        for k in range(number_of_subplots):
            results_current_step_real_water = result_multiindex.xs(k+1, level=1)/total_water_available*100
            results_current_step_virtual_water = result_multiindex.xs(k+1, level=1)/init_water[str(k+1)]*100                   
            ax[k].plot(results_current_step_real_water.index, results_current_step_real_water['water_available'].values, color='blue', alpha=0.1)
            ax[k].plot(results_current_step_virtual_water.index, results_current_step_virtual_water['virtual_water_available'].values, color='black', alpha=0.2)
        
        

def agents_position(agents_results):
    farmers_results = agents_results[agents_results['Type'] == 'farmer']
    farmers_latest = farmers_results.xs(20, level=0)
    plt.hist(farmers_latest['Position'], bins=15)
    plt.ylabel('N. of farmers')
    plt.xlabel('Section')
    position = np.linspace(0, 10000, num=15)
    plt.xticks(position, (list(range(1,16))))
    plt.rcParams["figure.dpi"] = 300