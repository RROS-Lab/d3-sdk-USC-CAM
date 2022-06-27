"""
Author: Anirban Mukherjee

This code is a simulation of Structured Action Prediction with ideas borrowed from the letter
"Structured Action Prediction for Tele-Operation in Open Worlds" by Naughton et al. We use the methodology
in this paper to demonstrate an action predictor based on an input sequence of actions.
"""

## Imports
import numpy as np
import torch
import torch.nn as nn
import csv

## Read Input, save as a Vector
## Read Files
input_file = open('Input.csv')
input_raw = csv.reader(input_file)
input_data = []

for line in input_raw:
    input_data.append(int(line[0]))
input_file.close()

## Context Vector
d = 5
context_vectors = []

# Creat Correct Prediction
correct_predictions = []

for i in range(5, len(input_data)):
    context_vectors.append([input_data[i-1], input_data[i-2], input_data[i-3], input_data[i-4], input_data[i-5]])
    prediction_array = np.zeros(10)
    prediction_array[input_data[i]] = 1
    correct_predictions.append(prediction_array)

# Initialize Neural Net Using Torch Sequential 
model = nn.Sequential(
    nn.Linear(5, 10),
    nn.ReLU(),
    nn.Linear(10,30),
    nn.ReLU(),
    nn.Linear(30,10),
    nn.Softmax(dim = 0)
)

# Initialize Loss Function
loss_function = nn.CrossEntropyLoss()

# Training
for epoch in range(20):
    print("Epoch #" + str(epoch))
    total_loss = 0.0

    for i in range(len(context_vectors)):
        output = model(torch.FloatTensor([context_vectors[i]]))
        loss = loss_function(output, torch.FloatTensor([correct_predictions[i]]))
        loss.backward()

        print("total loss is " + str(total_loss))

print("finished training")