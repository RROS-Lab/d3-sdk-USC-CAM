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
import matplotlib.pyplot as plt

## Read Input, save as a Vector
## Read Files
input_file = open('Input.csv')
input_raw = csv.reader(input_file)
input_data = []

for line in input_raw:
    input_data.append(int(line[0]))
input_file.close()

## Context Vector
d = 7
context_vectors = []

# Creat Correct Prediction
correct_predictions = []

for i in range(d, len(input_data)):
    context_vectors.append(input_data[i-d:i])
    prediction_array = np.zeros(10)
    prediction_array[input_data[i]] = 1
    correct_predictions.append(prediction_array)

# Initialize Neural Net Using Torch Sequential 
model = nn.Sequential(
    nn.Linear(d, 10),
    nn.ReLU(),
    nn.Linear(10,30),
    nn.ReLU(),
    nn.Linear(30,15),
    nn.ReLU(),
    nn.Linear(15,10),
    nn.Softmax(dim = 1)
)

# Initialize Loss Function and Optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

loss_scale = []
epoch_scale = []

for epoch in range(200):
    #print("Epoch #" + str(epoch))
    total_loss = 0.0

    for i in range(len(context_vectors)):
        optimizer.zero_grad()
        output = model(torch.FloatTensor([context_vectors[i]]))
        loss = loss_function(output, torch.FloatTensor([correct_predictions[i]]))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # for plotting 
    loss_scale.append(total_loss)
    epoch_scale.append(epoch)


print("total loss is " + str(total_loss))

print("finished training")

# plot loss
plt.plot(loss_scale, epoch_scale, 'r')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

#Testing
print("Beginning Testing")

correct = 0 
data_size = len(context_vectors)

for i in range(data_size):
    guess_raw = model(torch.Tensor([context_vectors[i]]))
    guess = torch.argmax(guess_raw)
    correct_prediction = np.argmax(correct_predictions[i])

    if (guess == correct_prediction):
        correct += 1

accuracy = correct/data_size

print("accuracy after evaluation is " + str(accuracy))

