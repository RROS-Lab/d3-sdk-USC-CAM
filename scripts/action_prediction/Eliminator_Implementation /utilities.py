from hashlib import new
from statistics import mean
import numpy as np
import torch
import torch.nn as nn
import csv
import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans

"""
This file defines the functions and classes which will be used in main.py for 
elimination assisted/prediction of actions
"""

## Create Context Vectors and Correct Predictions
"""
Create the background context vectors
"""
## Open the data file for reading
input_file = open('eliminator_data.csv')
input_raw = csv.reader(input_file)
input_data = []

for line in input_raw:
    input_data.append(int(line[0]))
input_file.close()

## Create context vector
d = 7 ## Context Vector Size
num_actions = 14 ## Specify the number of Action in the action vocabulary 

training_context_vectors_eliminator = [] ## List of training context vectors
evaluation_context_vectors_eliminator = [] ## List of evaluation context vectors

## Create Labels for Training and Evaluation
training_predictions_eliminator = []  ## List of labels for training
raw_training_predictions_eliminator = [] ## List of the actions corresponding to each label for training

evaluation_predictions_eliminator = [] ## List of labels for evaluation 
raw_evaluation_predictions_eliminator = [] ## List of the actions corresponding to each label for evaluation

for i in range(d, len(input_data)):
    new_cv = np.zeros(num_actions) 
    count = 1
    d_vec = input_data[i-d:i]

    for j in range(len(d_vec)):
        cv_index = d_vec[j]
        new_cv[cv_index] += count
        count += 1
    
    ## Populate Predictions
    prediction_array = np.zeros(num_actions)
    prediction_array[input_data[i]] = 1

    new_cv = new_cv/(np.linalg.norm(new_cv))

    # For Training 
    if i <= 160:
        training_context_vectors_eliminator.append(new_cv)
        training_predictions_eliminator.append(prediction_array)
        raw_training_predictions_eliminator.append(input_data[i])
    
    # For Evaluation
    else:
        evaluation_context_vectors_eliminator.append(new_cv)
        evaluation_predictions_eliminator.append(prediction_array)
        raw_evaluation_predictions_eliminator.append(input_data[i])

"""
Create the user-specific context vectors
"""
## Open the data file for reading
input_file = open('predictor_data.csv')
input_raw = csv.reader(input_file)
input_data = []

for line in input_raw:
    input_data.append(int(line[0]))
input_file.close()

## Create context vector
d = 7 ## Context Vector Size
num_actions = 14 ## Specify the number of Action in the action vocabulary 

training_context_vectors_predictor = [] ## List of training context vectors
evaluation_context_vectors_predictor = [] ## List of evaluation context vectors

## Create Labels for Training and Evaluation
training_predictions_predictor = []  ## List of labels for training
raw_training_predictions_predictor = [] ## List of the actions corresponding to each label for training

evaluation_predictions_predictor = [] ## List of labels for evaluation 
raw_evaluation_predictions_predictor = [] ## List of the actions corresponding to each label for evaluation

for i in range(d, len(input_data)):
    new_cv = np.zeros(num_actions) 
    count = 1
    d_vec = input_data[i-d:i]

    for j in range(len(d_vec)):
        cv_index = d_vec[j]
        new_cv[cv_index] += count
        count += 1
    
    ## Populate Predictions
    prediction_array = np.zeros(num_actions)
    prediction_array[input_data[i]] = 1

    new_cv = new_cv/(np.linalg.norm(new_cv))

    # For Training 
    if i <= 25:
        training_context_vectors_predictor.append(new_cv)
        training_predictions_predictor.append(prediction_array)
        raw_training_predictions_predictor.append(input_data[i])
    
    # For Evaluation
    else:
        evaluation_context_vectors_predictor.append(new_cv)
        evaluation_predictions_predictor.append(prediction_array)
        raw_evaluation_predictions_predictor.append(input_data[i])

## Create Eliminator and Predictor Classes
"""
Create both Neural Networks using Torch.Sequential
"""

eliminator_model = nn.Sequential(
    nn.Linear(num_actions, 23),
    nn.ReLU(),
    nn.Linear(23,num_actions)
)

predictor_model = nn.Sequential(
    nn.Linear(2*num_actions,23),
    nn.ReLU(),
    nn.Linear(23,30),
    nn.ReLU(),
    nn.Linear(30, num_actions),
    #nn.Softmax(dim = 1)
)

## Define Neural Net Loss Functions and Optimizers
loss_function_eliminator = nn.MSELoss()
optimizer_eliminator = torch.optim.SGD(eliminator_model.parameters(), lr = 0.05)

#loss_function_predictor = nn.CrossEntropyLoss()
loss_function_predictor = nn.MSELoss()
optimizer_predictor = torch.optim.SGD(predictor_model.parameters(), lr = 0.02)

## Define Training Function (with plot)
def make_tensor(array):
    return torch.FloatTensor([array])

""" 
Eliminator Training Function
"""
def eliminator_training(context_vectors, labels, epochs, plot=True):

    # Axes for Plotting
    loss_scale = []
    epoch_scale = []

    # Training Loop 
    for i in range(epochs):

        # Print the number of epochs and initialize loss at every epoch to zero 
        print("Epoch #" + str(i))
        total_loss = 0.0

        # Train on each context vector
        for j in range(len(context_vectors)):

            ## Set Gradient for Optimizer equal to Zero
            optimizer_eliminator.zero_grad()

            ## Forward Pass
            eliminator_input = context_vectors[j]
            output = eliminator_model(torch.FloatTensor([eliminator_input]))

            ## Calculate Loss and Backpropagate using training labels
            loss = loss_function_eliminator(output, torch.FloatTensor([labels[j]]))
            loss.backward()
            optimizer_eliminator.step()
            total_loss += loss.item()

        ## For Plotting
        loss_scale.append(total_loss)
        epoch_scale.append(i)

    print("finished training")

    if plot: 
        ## Plot Loss
        plt.plot(epoch_scale, loss_scale, 'r')
        plt.xlabel('epochs')
        plt.ylabel('loss')  
        plt.show()

"""
Predictor Training Function and helper
"""

def bayesian_merge(tensor1, tensor2):

    ## Method for combining two tensors using bayesian update
    new_array = []

    vector1 = tensor1.cpu().detach().numpy()[0]
    vector2 = tensor2.cpu().detach().numpy()[0]

    for i in range(len(vector1)):
        new_array.append(vector1[i])

    for i in range(len(vector2)):
        new_array.append(vector2[i])
    
    return make_tensor(new_array)


def filter_eliminated(input, elimination_probability):

    ## Input is a tensor
    raw_vector = input.cpu().detach().numpy()[0]
    new_vector = []

    ## Go through the vector, convert anything below the probability to 0, else 1
    for i in range(len(raw_vector)):
        if raw_vector[i] <= elimination_probability:
            new_vector.append(0)
        else:
            new_vector.append(1)

    return torch.FloatTensor([new_vector])
    
def predictor_training(context_vectors, labels, epochs, elimination_probability, plot=True):

    # Axes for Plotting
    loss_scale = []
    epoch_scale = []

    # Training Loop 
    for i in range(epochs):

        # Print the number of epochs and initialize loss at every epoch to zero 
        print("Epoch #" + str(i))
        total_loss = 0.0

        # Train on each context vector
        for j in range(len(context_vectors)):

            ## Set Gradient for Optimizer equal to Zero
            optimizer_predictor.zero_grad()

            ## Forward Pass through eliminator network 
            eliminator_input = context_vectors[j]
            eliminator_output = eliminator_forward(eliminator_input)
            eliminator_output = filter_eliminated(eliminator_output, elimination_probability)

            ## Concatenate output of eliminator network with context vector, forward pass through predictor
            predictor_input = bayesian_merge(make_tensor(eliminator_input), eliminator_output)
            predictor_output = predictor_model(predictor_input)
            
            ## Calculate Loss and Backpropagate using training labels
            loss = loss_function_predictor(predictor_output, make_tensor(labels[j]))
            loss.backward()
            optimizer_predictor.step()
            total_loss += loss.item()

        ## For Plotting 
        loss_scale.append(total_loss)
        epoch_scale.append(i)

    print("finished training")

    if plot: 
        ## Plot Loss
        plt.plot(epoch_scale, loss_scale, 'r')
        plt.xlabel('epochs')
        plt.ylabel('loss')  
        plt.show()

## Define Output Function
def eliminator_forward(input):
    return eliminator_model(torch.FloatTensor([input]))

def predictor_forward(input):
    return predictor_model(torch.FloatTensor([input]))

## Define Evaluation Functions
def evaluate_predictor(context_vectors, raw_labels, elimination_probability):

    correct = 0

    for i in range(len(context_vectors)):
        
        ## Store the current context vector in a variable
        evaluation_input = context_vectors[i]
        #print(evaluation_input)

        ## Forward Pass the Context Vector
        eliminator_output = filter_eliminated((eliminator_forward(evaluation_input)), elimination_probability)
        bayesian_input = bayesian_merge(make_tensor(evaluation_input), eliminator_output)
        #print(bayesian_input)
        guess_raw = predictor_model(bayesian_input)

        ## Argmax the output layer
        guess = torch.argmax(guess_raw)

        print(guess)

        ## Take the argmax of the prediction labels in order to validate
        correct_prediction = raw_labels[i]

        #print(correct_prediction)

        if (guess == correct_prediction):
            correct += 1
    
    accuracy = correct/len(context_vectors)

    print("prediction accuracy after training is " + str(accuracy))

"""
Train
"""
#if __name__ == "main":
eliminator_training(training_context_vectors_eliminator, training_predictions_eliminator, 200)
predictor_training(training_context_vectors_predictor, training_predictions_predictor, 2000, 0.05)

"""
Evaluate
"""
evaluate_predictor(training_context_vectors_predictor, raw_training_predictions_predictor, 0.05)
evaluate_predictor(evaluation_context_vectors_predictor, raw_evaluation_predictions_predictor, -40000)