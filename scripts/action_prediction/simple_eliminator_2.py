"""
Author: Anirban Mukherjee

This is a simple eliminator network designed to track the lowest observed probability of recommended
actions in our action vocabulary.
"""

## Imports
from hashlib import new
from statistics import mean
import numpy as np
import torch
import torch.nn as nn
import csv
import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans

## Read Input, save as a Vector
input_file = open('eliminator_data.csv')
input_raw = csv.reader(input_file)
input_data = []

for line in input_raw:
    input_data.append(int(line[0]))
input_file.close()

## Context Vector
d = 7
num_actions = 14
context_vectors = []
evaluation_data = []

## Create Correct Predictions
correct_predictions_training = []
training_predictions_raw = []

for i in range(d, len(input_data)):

    new_cv = np.zeros(num_actions)
    count = 1
    d_vec = input_data[i-d:i]

    for j in range(len(d_vec)):
        cv_index = d_vec[j]
        new_cv[cv_index] += (count + 1)
        count += 1
    
    ## Populate Predictions
    prediction_array = np.zeros(num_actions)
    prediction_array[input_data[i]] = 1

    context_vectors.append(new_cv)
    correct_predictions_training.append(prediction_array)
    training_predictions_raw.append(input_data[i])

for i in range(len(context_vectors)):
    print("context vector")
    print(context_vectors[i])
    #print(correct_predictions_training[i])


## Initialize Neural Net Using Torch Sequential 
model = nn.Sequential(
    nn.Linear(num_actions, 23),
    nn.ReLU(),
    # nn.Linear(23, 30),
    # nn.ReLU(),
    # nn.Linear(30, 40),
    # nn.ReLU(),
    # nn.Linear(40, 23),
    # nn.ReLU(),
    nn.Linear(23,num_actions),
    #nn.Softmax(dim = 1)
)

## Initialize Loss Function and Optimizer
#loss_function = nn.CrossEntropyLoss()
loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.005)

loss_scale = []
epoch_scale = []
numpy_log_inputs = []
numpy_log_outputs = []

# evaluation_data_inputs = []
# evaluation_data_outputs = []

print("started training")

for epoch in range(200):

    print("Epoch #" + str(epoch))
    total_loss = 0.0

    for i in range(len(context_vectors)):

        ## Set Gradient for Optimizer equal to Zero
        optimizer.zero_grad()

        ## Forward Pass
        eliminator_input = context_vectors[i]
        output = model(torch.FloatTensor([eliminator_input]))

        ## Log Eliminator Input and Output to CSV
        if epoch == 199:
            print("I'm Logging")
            numpy_log_inputs.append(eliminator_input)
            numpy_log_outputs.append(output.cpu().detach().numpy()[0])

        ## Calculate Loss and Backpropagate using training labels
        loss = loss_function(output, torch.FloatTensor([correct_predictions_training[i]]))

        if epoch == 199:
            print("output is")
            print(torch.argmax(output))
            print("label is")
            print(np.argmax(correct_predictions_training[i]))

        ## Backprop
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    ## For Plotting 
    loss_scale.append(total_loss)
    epoch_scale.append(epoch)

print("total loss is " + str(total_loss))

print("finished training")

## Plot Loss
plt.plot(epoch_scale, loss_scale, 'r')
plt.xlabel('epochs')
plt.ylabel('loss')  
plt.show()

np.savetxt("eliminator_inputs.csv", numpy_log_inputs)
np.savetxt("eliminator_outputs.csv", numpy_log_outputs)

# Save labels for logging
np.savetxt("training_predictions", training_predictions_raw)

for i in range(len(numpy_log_outputs)):
    print(np.argmax(numpy_log_outputs[i]))

correct = 0 
incorrect_eliminations = 0
elimination_probability = 0.03
training_number_eliminated_array = []
data_size = len(context_vectors)

## Validate Prediction and Elimination Accuracy 
for i in range(data_size):
    
    ## Store the current context vector in a variable
    evaluation_input = context_vectors[i]

    ## Forward Pass the Context Vector
    guess_raw = model(torch.Tensor([evaluation_input]))

    ## Argmax the output layer
    guess = torch.argmax(guess_raw)

    ## Take the argmax of the prediction labels in order to validate
    correct_prediction = np.argmax(correct_predictions_training[i])

    ## Append the data for inputs and outputs
    # evaluation_data_inputs.append(evaluation_input)
    # evaluation_data_outputs.append(guess_raw.cpu().detach().numpy()[0])

    if (guess == correct_prediction):
        correct += 1


    ## Check what actions have been eliminated
    eliminated_actions = 0 
    guess_raw_detached = guess_raw.cpu().detach().numpy()[0]

    for i in range(len(guess_raw_detached)):

        ## Find out how many actions were eliminated
        if guess_raw_detached[i] <= elimination_probability:
            eliminated_actions += 1

            ## Find out if the correct action was eliminated 
            if i == correct_prediction:
                incorrect_eliminations += 1

    training_number_eliminated_array.append(eliminated_actions)

accuracy = correct/data_size
average_eliminations = mean(training_number_eliminated_array)

print("prediction accuracy after training is " + str(accuracy))

print("average number of actions eliminated under " + str(elimination_probability) + " is " + str(average_eliminations))

print("percent of context vectors where incorrect action was eliminated is " + str(incorrect_eliminations/data_size))