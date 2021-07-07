# -*- coding: utf-8 -*-
import cleanse

import pandas as pd
from flask import Flask, render_template, request
from flask import jsonify
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
MAX_SEQUENCE_LENGTH = 250
MAX_NB_WORDS = 50000
app = Flask(__name__)


exit_msgs = ['bye', 'goodbye', 'good bye', 'ok bye']


model = load_model('model_v2.h5')
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

id2classes = {0: 'Bees/Venomous Animals',
 1: 'Blocking and isolation of energies',
 2: 'Fall',
 3: 'Hazards during Electrical installations',
 4: 'Injuries/Cuts',
 5: 'Manual Tools',
 6: 'NA',
 7: 'Others',
 8: 'Power lock',
 9: 'Pressed',
 10: 'Pressurized Systems/ Chemical Substances',
 11: 'Projections of fragments',
 12: 'Suspended Loads',
 13: 'Vehicles and Mobile Equipment'}

standard_to = StandardScaler()
with open('./other_files/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

@app.route("/predict", methods=['GET'])
def predict():
    if request.method == 'GET':
        Desc = request.args.get('msg')
        Desc = Desc.lower()
        if Desc in exit_msgs:
            return {"pred" : "I hope I was able to help you. Bye!", "active" : 0}
        elif 'thank' in Desc:
            return {'pred' : "Great! Is there anything else you need help with?", 'active': 1}

        chatbot = cleanse.Chatbot(model, tokenizer, id2classes, 0.3)
        response, mean_accident_level, mean_potential_accident_level = chatbot.chatbot_response(Desc) 

        return {"pred": response, "active":1, "accident_level": mean_accident_level, "potential_accident_level": mean_potential_accident_level}
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)