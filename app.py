import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import pickle


app = Flask(__name__, static_url_path='/static')
data = pd.read_csv('Cleaned_data_house.csv')
pipe = pickle.load(open("model.pkl", "rb"))


@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    districts = sorted(data['district'].unique())
    return render_template('House_Shylesh.html', locations=locations, districts=districts)


@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    sqft = request.form.get('total_sqft')
    dist = request.form.get('district')
    input = pd.DataFrame([[location, sqft, bath, bhk, dist]], columns=['location', 'total_sqft', 'bath', 'bhk', 'district'])
    prediction = pipe.predict(input)[0]
    return str(np.round(prediction,2))

