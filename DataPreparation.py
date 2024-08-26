# -*- coding: utf-8 -*-
"""

@author: machadouyang
"""

import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import scipy.stats as ss
import networkx as nx

def get_evenly_divided_values(value_to_be_distributed, times):
    """
    Divide a number into (almost) equal whole numbers
    
    Args:
        value_to_be_distributed: number to be divided
        times: Number of integer values to divide
    """ 
    return [value_to_be_distributed // times + int(x < value_to_be_distributed % times) for x in range(times)]

def generate_edges_linear_graph(number_of_sections = 10, number_of_nodes=25):
    """
    Create a linear graph with 'section' attribute (almost) evenly spaced
    according to number of edges
    
    Args:
        number_of_sections: Number of sections to devide nodes
        number_of_nodes: Number of nodes (farmers possible positions)
    """ 
    
    n_nodes_per_section = get_evenly_divided_values(number_of_nodes, number_of_sections)
    sections_list = []
    for i, v in enumerate(n_nodes_per_section):
        for j in range(v):
            sections_list.append(i+1)   
    sections = {i+1: {'section': v} for i, v in enumerate(sections_list)}
    edges = []
    for x in range(1, number_of_nodes):
        edges.append((x,x+1))
    linear_graph = nx.Graph()
    linear_graph.add_edges_from(edges)
    nx.set_node_attributes(linear_graph, sections)
    # nx.draw_networkx(linear_graph)
    return linear_graph

def prepare_output_structure():
    agents_df_columns = ["id", "step", "type", "position", "water_need", "water_withdrew", "contract", "revenue", "farm_area", "chosen_crop"]
    model_df_columns = ["step", "section", "water_available", "virtual_water_available"]
    agents_results = pd.DataFrame(columns= agents_df_columns)
    model_results = pd.DataFrame(columns= model_df_columns)
    return agents_results, model_results