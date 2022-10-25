"""
Authors: Anir and JaeHyun

This script reads the csv files corresponding to the robot detecting aruco_markers and the current
desired action for the robot. It then sends a command to the D3. 
"""

## Imports
import csv
from DRDoubleSDK import DRDoubleSDK as double
import sys
import socket
import os # Just for exit()
import time

# d3 = double.DRDoubleSDK()
## Function for driving to a certain aruco marker
def drive_to(x,z):
    
    # Proportion constant
    Kp = 0
    
    # Calculate error (desired - actual)
    x_error = 0 - x
    z_error = 2 - z

    # Send command to D3
    send_speed(Kp*z, Kp*x)

## Function for sending velocity commands to the robot
def send_speed(straight_speed, turn_speed):
    # Check if speed values are -1.0 ~ 1.0
    if(straight_speed >= -1 and straight_speed <= 1 and turn_speed >= -1 and turn_speed <= 1):
        # Send speed valeus to D3
        # d3.sendCommand('navigate.drive',  { "throttle": straight_speed, "turn": turn_speed, "powerDrive": "false" });

        # Send Speed for each 200ms
        # TODO

        print("sent speed to d3: " + str(straight_speed) + "," + str(turn_speed))
    else:
        print("Error: Check speed values. (-1.0 ~ 1.0)")

## Function to disable navigation commands on robot
def cancel_nav():
    pass
    # TODO send navigation cancellation

## Initializing D3 commands
# d3.sendCommand('navigate.enable');

while True: 

    # Read backend_csv.csv, populate current_command
    current_command = open('CSV_txt/sidebar_button_num.csv').readlines()
    if current_command:
        current_command = current_command[0]
 
    # Store data from client
    # Data will be a list of string that looks like [id_1, x1, y1, z1, id_2, x2, y2, z2 ...]
    data_from_client = open('CSV_txt/data_from_client.txt').readlines()
    # Making form of [ids, tvecs]
    if data_from_client:
        data_from_client[0] = data_from_client[0].replace("""b'""", '')
        data_from_client[0] = data_from_client[0].replace("""'""", '')
        data_from_client = data_from_client[0].split()
        print(data_from_client)

    if current_command == "0":
        # Cancel Navigation
        cancel_nav()
        print("no command detected")
   
    else: 
        # if an object 1 is detected
        saw_id = False
        index_of_detected = 0
        
        # Look through detection data for the id asscociated with current command
        for i in range(len(data_from_client) - 4):
            # If the current value is equal to the current command, break and set saw_id = True, save index of id
            if data_from_client[i] == current_command:
                saw_id = True
                index_of_detected = i
                break
            # Else check the next id
            else:
                i = i + 4

        if saw_id:
            # Drive to the id's x and z values
            print("ID detected, driving to aruco marker")
            drive_to(data_from_client[index_of_detected + 1], data_from_client[index_of_detected + 3])
        else:
            print("specified ID undetected, rotating in place until id is found")
            # Rotate in place 
            send_speed(0, 1)
    
    time.sleep(0.02)



