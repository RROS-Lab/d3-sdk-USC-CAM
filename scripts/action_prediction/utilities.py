"""
Author: Omey Manyar
Email ID: manyar@usc.edu

This file consists of regular utility functions for following tasks:
1. Parsing and Storing i/o data
2. Similarity and Activation functions used for training
"""

## Imports
import copy
import numpy as np
import csv
import matplotlib.pyplot as plt



####Function to analyze input context vector and types of actions taken###
def evaluate_input_data(context_vector, labeled_output, verbose = False):

    data_information_dict = {}      #Dictionary that takes input as the aray

    unique_context_vectors = np.unique(context_vector, axis=0)

    if(verbose):
        print("Size of original Context vectors: ", len(context_vector))
        print("Unique Vectors: \n", unique_context_vectors)
        print("size of unique context vectors: ", len(unique_context_vectors))
        print("Length of the prediction vector: ", len(labeled_output))

    data_information_dict = {}

    for i in range(len(context_vector)):
        current_key = str(context_vector[i])
        
        if(current_key in data_information_dict.keys()):
            current_prediction_arr = copy.deepcopy(data_information_dict[current_key])
            current_prediction_arr.append(labeled_output[i])
            data_information_dict[current_key] = current_prediction_arr
        else:
            data_information_dict[current_key] = [labeled_output[i]]
        
    if(verbose):
        print("Dictionary of Context Vectors and labelled actions: ", data_information_dict)

    return data_information_dict

###Function to save data as a csv file

def save_context_vec_data(data_information_dict, filename = 'context_vector_diversity.csv', verbose = False):

    rows = []
    for keys in data_information_dict.keys():
        value = copy.deepcopy(data_information_dict[keys])
        current_row = [keys, value]
        rows.append(current_row)

    
    fields = ['Context Vectors', 'Action Array']
    
    if(verbose):
        print("Data to save: ", rows)
        print('fields: ', fields)
        print('rows: ', rows)
            
    with open(filename, 'w') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile) 
            
        # writing the fields 
        csvwriter.writerow(fields) 
            
        # writing the data rows 
        csvwriter.writerows(rows)


    return