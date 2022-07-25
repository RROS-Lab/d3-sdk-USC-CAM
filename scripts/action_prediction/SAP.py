"""
Author: Anirban Mukherjee

This code is a simulation of Structured Action Prediction with ideas borrowed from the letter
"Structured Action Prediction for Tele-Operation in Open Worlds" by Naughton et al. We use the methodology
in this paper to demonstrate an action predictor based on an input sequence of action parameter combinations. 
"""

## Imports
import numpy as np
import torch
import torch.nn as nn
import csv
import matplotlib.pyplot as plt

"""
The input file to this script (Input_Action.csv) contains a series of discrete action parameter combos each encoded to 
a specific scalar value between 0 and 9 inclusively. The network should utilize this series of actions in order
to train on the most likely action given a brief history of previous actions. 
"""

## Read Input, save as a Vector
input_file = open('Training.csv')
input_raw = csv.reader(input_file)
input_data = []

for line in input_raw:
    input_data.append(int(line[0]))
input_file.close()

"""
Encoding Actions

Driving (Action 1)
1: Drive to Assembly Table
2: Drive to Toolbox
3: Drive to Parts Area

Scan for Tools (Action 2)
4: Scan for Screwdriver
5: Scan for Hammer
6: Scan for Chain Breaker

Scan for Parts (Action 3)
7: Scan for Wheels
8: Scan for Chain 
9: Scan for Brakes
10: Scan for Handlebars
11: Scan for Seat
12: Scan for Frame

Laser Pointer (Action 4)
13: Laser On
14: Laser Off
"""

# Create a function that converts encoding to action number
def isAction(action_value):
    if action_value >= 1 and action_value <= 3:
        return 1
    elif action_value >= 4 and action_value <= 6:
        return 2
    elif action_value >= 5 and action_value <= 10:
        return 3
    elif action_value >= 11 and action_value <= 12:
        return 4

## Specify context vector size
d = 8 

## Initialize Action Network 
ActionNet = nn.Sequential(
    nn.Linear(d, 10),
    nn.ReLU(),
    nn.Linear(10,30),
    nn.ReLU(),
    nn.Linear(30,15),
    nn.ReLU(),
    nn.Linear(15,10),
    nn.Softmax(dim = 1)
)

def ActionScorer(input):
    output = ActionNet(input)
    output = isAction(output)
    return output

## Initialize Parameter Networks 
ParamNetDrive = nn.Sequential(
    nn.Linear(d, 10),
    nn.ReLU(),
    nn.Linear(10,30),
    nn.ReLU(),
    nn.Linear(30,15),
    nn.ReLU(),
    nn.Linear(15,3),
    nn.Softmax(dim = 1)
)

def ParamScorerDrive(input):
    output = ParamNetDrive(input)
    return output

ParamNetTools = nn.Sequential(
    nn.Linear(d, 10),
    nn.ReLU(),
    nn.Linear(10,30),
    nn.ReLU(),
    nn.Linear(30,15),
    nn.ReLU(),
    nn.Linear(15,3),
    nn.Softmax(dim = 1)
)

def ParamScorerTools(input):
    output = ParamNetTools(input)
    return output

ParamNetParts = nn.Sequential(
    nn.Linear(d, 10),
    nn.ReLU(),
    nn.Linear(10,30),
    nn.ReLU(),
    nn.Linear(30,15),
    nn.ReLU(),
    nn.Linear(15,6),
    nn.Softmax(dim = 1)
)

def ParamScorerParts(input):
    output = ParamNetParts(input)
    return output

ParamNetPointer = nn.Sequential(
    nn.Linear(d, 10),
    nn.ReLU(),
    nn.Linear(10,30),
    nn.ReLU(),
    nn.Linear(30,15),
    nn.ReLU(),
    nn.Linear(15,2),
    nn.Softmax(dim = 1)
)

def ParamScorerPointer(input):
    output = ParamNetPointer(input)
    return output
"""add eliminator net"""

## Create Cost Function
def cost_function(context_vector):

    action_score_vector = ActionScorer(context_vector)
    param__score_vector_drive = ParamScorerDrive(context_vector)
    param__score_vector_tool = ParamScorerTools(context_vector)
    param_score_vector_parts = ParamScorerParts(context_vector)
    param_score_vector_pointer = ParamScorerPointer(context_vector)
    
    cost_vector = []

    ## Drive costs
    cost_vector[0] = action_score_vector[0] + param__score_vector_drive[0]
    cost_vector[1] = action_score_vector[0] + param__score_vector_drive[1]
    cost_vector[2] = action_score_vector[0] + param__score_vector_drive[2]
    
    ## Tool Costs 
    cost_vector[3] = action_score_vector[1] + param__score_vector_tool[0]
    cost_vector[4] = action_score_vector[1] + param__score_vector_tool[1]
    cost_vector[5] = action_score_vector[1] + param__score_vector_tool[2]

    ## Part Costs 
    cost_vector[6] = action_score_vector[2] + param_score_vector_parts[0]
    cost_vector[7] = action_score_vector[2] + param_score_vector_parts[1]
    cost_vector[8] = action_score_vector[2] + param_score_vector_parts[2]
    cost_vector[9] = action_score_vector[2] + param_score_vector_parts[3]
    cost_vector[10] = action_score_vector[2] + param_score_vector_parts[4]
    cost_vector[11] = action_score_vector[2] + param_score_vector_parts[5]

    ## Laser Pointer Costs
    cost_vector[12] = action_score_vector[3] + param_score_vector_pointer[0]
    cost_vector[13] = action_score_vector[3] + param_score_vector_pointer[1]

    return cost_vector

## Create a function to get recommended action param combo given a context vector
def get_recommendation(context_vector):
    return np.argmax(get_recommendation(context_vector))

"""
For training, we use data from our training dataset and SVM loss on the function 
λimax(a^,ψ^)[Δ((a^,ψ^),(ai,ψi))+E(xi,a^,ψ^)-E(xi,ai,ψi)].+
"""

## Specify number of Epochs
Epochs = 200

def GetLambda(input):
    """
    TODO
    """

## Specify loss for SVM Loss Function 
def SVMLossFunction(output, truths):
    batch_size = output.size()[0]
    
    # Calculate Lambda_i

    # Calculate SVM Loss 

    """
    TODO 
    """

## Specify Optimizers for each neural network
action_optimizer = torch.optim.SGD(ActionNet.parameters(), lr = 0.01)
param_loss_drive = torch.optim.SGD(ParamNetDrive.parameters(), lr = 0.01)
param_loss_tools = torch.optim.SGD(ParamNetTools.parameters(), lr = 0.01)
param_loss_parts = torch.optim.SGD(ParamNetParts.parameters(), lr = 0.01)
param_loss_pointer = torch.optim.SGD(ParamNetPointer.parameters(), lr = 0.01)


## Initialize Arrays for Plotting
loss_scale_action = []
loss_scale_drive = []
loss_scale_tools = []
loss_scale_parts = []
loss_scale_pointer = []

epoch_scale = []

for epoch in range(200):
    #print("Epoch #" + str(epoch))
    total_loss = 0.0

    for i in range(len(context_vectors)):
        optimizer.zero_grad()
        output = model(torch.FloatTensor([context_vectors[i]]))
        loss_action = loss_function(output, torch.FloatTensor([correct_predictions[i]]))
        loss_drive = 
        loss_tools = 
        loss_parts = 
        loss_pointer
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

