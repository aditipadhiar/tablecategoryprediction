# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 13:39:44 2023

@author: AAA
"""

from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model = pickle.load(open('trained_model.pkl', 'rb'))
cv = pickle.load(open('transform.pkl', 'rb'))
@app.route('/', methods=['GET'])
def Home():
    #return render_template('index.html')
    return "Hello world!"

standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        Tablename = str(request.form['Tablename'])  
        data = [Tablename]
        vect = cv.transform(data).toarray()
        prediction=model.predict(vect)
        return str(prediction)
        #return render_template('index.html', prediction_text="The Category is: {}".format((prediction)))
        #return str(prediction_text="The Category is: {}".format((prediction)))

    #else:
        #return render_template('index.html')

if __name__=="__main__":
    response_API = requests.get('https://web-production-20ea.up.railway.app')
    print(response_API.text)
    app.run(debug=True)