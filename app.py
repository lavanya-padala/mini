from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
import cv2
import pickle
import imutils
import sklearn
from tensorflow.keras.models import load_model
# from pushbullet import PushBullet
import joblib
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input


# Loading Models
heart_model = joblib.load('logistic_regression_model.joblib')
diabetes_model = joblib.load('diabetes_model.joblib')
breastcancer_model = joblib.load('models/cancer_model.pkl')


app = Flask(__name__)
app.secret_key = "secret key"


########################### Routing Functions ########################################

@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/breastcancer')
def breast_cancer():
    return render_template('breastcancer.html')

@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')


@app.route('/heartdisease')
def heartdisease():
    return render_template('heartdisease.html')



@app.route('/resultd', methods=['POST'])
def resultd():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        gender = request.form['gender']
        pregnancies = float(request.form['pregnancies'])
        glucose = float(request.form['glucose'])
        bloodpressure = float(request.form['bloodpressure'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        diabetespedigree = float(request.form['diabetespedigree'])
        age = float(request.form['age'])
        skinthickness = float(request.form['skin'])
        pred = diabetes_model.predict(
            [[pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigree, age]])
        # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour Diabetes test results are ready.\nRESULT: {}'.format(firstname,['NEGATIVE','POSITIVE'][pred]))
        return render_template('resultd.html', fn=firstname, ln=lastname, age=age, r=pred, gender=gender)


@app.route('/resultbc', methods=['POST'])
def resultbc():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        gender = request.form['gender']
        age = request.form['age']
        cpm = request.form['concave_points_mean']
        am = request.form['area_mean']
        rm = request.form['radius_mean']
        pm = request.form['perimeter_mean']
        cm = request.form['concavity_mean']
        pred = breastcancer_model.predict(
            np.array([cpm, am, rm, pm, cm]).reshape(1, -1))
        # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour Breast Cancer test results are ready.\nRESULT: {}'.format(firstname,['NEGATIVE','POSITIVE'][pred]))
        return render_template('resultbc.html', fn=firstname, ln=lastname, age=age, r=pred, gender=gender)

@app.route('/resulth', methods=['POST'])
def resulth():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        age = float(request.form['age'])
        gender = int(request.form['gender'])
        cholestrol = float(request.form['cholestrol'])
        mhra = float(request.form['mhra'])
        eia = int(request.form['eia'])
        bp = float(request.form['bp'])
        tcp = int(request.form['tcp'])
        fbs = int(request.form['fbs'])
        thal = int(request.form['thal'])
        ekg = int(request.form['ekg'])
        st = int(request.form['st'])
        vf = int(request.form['vf'])
        std = float(request.form['std'])
        print(np.array([age,gender,tcp,bp,cholestrol,fbs,ekg,mhra,eia,std,st,vf,thal]).reshape(1, -1))
        pred = heart_model.predict(
            np.array([age,gender,tcp,bp,cholestrol,fbs,ekg,mhra,eia,std,st,vf,thal]).reshape(1, -1))
        # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour Diabetes test results are ready.\nRESULT: {}'.format(firstname,['NEGATIVE','POSITIVE'][pred]))
        return render_template('resulth.html', fn=firstname, ln=lastname, age=age, r=pred, gender=gender)


# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response


if __name__ == '__main__':
    app.run(debug=True)
