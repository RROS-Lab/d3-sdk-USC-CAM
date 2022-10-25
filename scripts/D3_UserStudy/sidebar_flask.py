"""
Authors: Jaehyun and Anir

This code creates a sidebar.html page, reads the buttons pressed, and
populates command.csv
"""

# Imports
from crypt import methods
from flask import Flask, render_template, request, Response
import csv

# Create Flask object
app = Flask(__name__)

# Open Sidebar.html in localhost
@app.route('/', methods = ["GET", "POST"])
def sidebar():
    # Make variable that contains the number clicked from sidebar.html
    if request.method == "POST":
        user_order = request.form.get("go")
        user_order = request.form.get("do")
        # Open the csv file in write mode
        with open('CSV_txt/sidebar_button_num.csv', 'w') as f:
            # Create the csv file writer
            writer = csv.writer(f)
            # Write a row to the csv file
            writer.writerow(user_order)

    return render_template('sidebar.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0")
