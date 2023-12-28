import os
from socket import herror
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from werkzeug.utils import secure_filename
from flask import Flask, request, render_template, flash, redirect, url_for
import pickle
from predict import lstm_predict

app = Flask(__name__)

global filename
filename = ""

app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif', 'csv'])
def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("index.html", xx= -1)
 

@app.route('/index')
def index1():
   return render_template('index.html')


@app.route('/home')
def home():
   return render_template('index.html')


@app.route('/register',methods = ['POST','GET'])
def registration():
	return render_template('register.html')


@app.route('/login',methods = ['POST','GET'])
def login():
    return render_template('login.html')


@app.route('/pricing.html')
def pricing():
   return render_template('pricing.html')

@app.route('/features.html')
def features():
   return render_template('features.html')



@app.route('/contact.html')
def contact():
   return render_template('contact.html')



@app.route('/blog.html')
def blog():
   return render_template('blog.html')

@app.route('/index.html')
def index2():
   return render_template('index.html')


 

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('upload.html', error='No file part')

        file = request.files['file']

        # If the user does not select a file, the browser also
        # submits an empty file without a filename
        if file.filename == '':
            return render_template('upload.html', error='No selected file')

        try:
            
            csv_file_path = file
            
            # Call your existing function to get the lists
            output_files, error_names, errors = lstm_predict(csv_file_path)
            
            # Render the HTML template with the lists
            return render_template('output_display.html', output_files=output_files, error_names=error_names, errors=errors)


        except Exception as e:
            return render_template('upload.html', error=str(e))

    return render_template('upload.html', error=None)


if __name__ == "__main__":
    app.run(debug=False)