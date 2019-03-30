#importing libraries
import os
import flask
from flask import Flask, render_template, request
import pickle
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.models import load_model

#creating instance of the class
app=Flask(__name__)
init=1
prediction={}
tokenizer=pickle.load(open('tokenizer.pickle','rb'))
model=load_model('my_model.h5')
model.load_weights('my_model_weights.h5')
model._make_predict_function()

#to tell flask what url shoud trigger the function index()
@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html',init=init,prediction=prediction)

#prediction function
def ValuePredictor(text):
    tokens=tokenizer.texts_to_sequences([text])
    maxlen=200
    tokens=pad_sequences(tokens,maxlen=maxlen)
    prediction=model.predict(tokens)
    prediction=prediction*10
    prediction=np.rint(prediction)
    temp={}
    temp1=['Toxic','Severe Toxic','Obscene','Threat','Insult','Identity Hate']
    for i in range(6):
        temp[temp1[i]]=prediction[0][i]
    return temp

@app.route('/result',methods = ['POST'])
def result():
    if request.method=='POST':
        init=0
        text=request.form['text']
        prediction=ValuePredictor(text)
        return render_template("index.html",init=init,prediction=prediction)
