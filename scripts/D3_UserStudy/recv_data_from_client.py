"""
Authors: Jaehyun and Anir

This script receives aruco marker IDs and Postion data, then sends it to D3 as a command form.
This code is a main server and opens sidebar.html in localhost.
"""
# Imports
import socket
import csv

HOST = "192.168.34.91"  # The server's hostname or IP address
PORT = 4000  # The port used by the server

# Creating Server Socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # Solving Arror Address already in use.
s.bind((HOST, PORT)) # Associate the socket with a specific network interface and port number
print("Listening to Client...")
s.listen() # Enables Server to Accept Connections
conn, addr = s.accept() # Accepting Connections from Clients and Communicate with them

with conn:# conn ==> Client Socket Object
    print(f"Connected by {addr}")
    

    while True: 
        data_from_client = conn.recv(1024) # Receives Any Data from Client
        print(data_from_client)
        # save data to txt
        # Open the txt file in write mode
        with open('CSV_txt/data_from_client.txt', 'w') as f:
            # Create the csv file writer
            writer = csv.writer(f)
            # Write a row to the csv file
            f.writelines(str(data_from_client))
            f.close()