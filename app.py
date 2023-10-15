import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import threading
from numpy import dot                                           
from numpy.linalg import norm 

app = Flask(__name__)

# Initialize model as None
loaded_model = None
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def load_model():
    print('start load model')
    global loaded_model
    # Load the model asynchronously
    model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"  # Example model URL
    loaded_model = hub.load(model_url)
    print('end load model')

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/prediction', methods=['POST'])
def prediction():
    
    global loaded_model
    
    if loaded_model is None:
        load_model()

    def embed(input):
        return loaded_model(input)    

    data = request.get_json()
    print(data)
    text1=data['text1']
    text2=data['text2']
    print(text1,text2)
    message = [text1,text2]
    print(message)
    message_embeddings = embed(message)
    print(message_embeddings)
    a = tf.make_ndarray(tf.make_tensor_proto(message_embeddings))
    cos_sim = dot(a[0], a[1])/(norm(a[0])*norm(a[1]))
    

    # Perform inference with the loaded model
    results = cos_sim
    print(results)
    
    return {"similarity score":str(results)}
@app.route('/predict',methods=['POST'])
def predict():
    global loaded_model
    
    if loaded_model is None:
        load_model()

    def embed(input):
        return loaded_model(input)  
    data=[str(x) for x in request.form.values()]
    print(data)
    text1=data[0]
    text2=data[1]
    message = [text1,text2]
    print(message)
    message_embeddings = embed(message)
    print(message_embeddings)
    a = tf.make_ndarray(tf.make_tensor_proto(message_embeddings))
    cos_sim = dot(a[0], a[1])/(norm(a[0])*norm(a[1]))
    results = cos_sim
    print(results)
    res={"similarity score":str(results)}
    return render_template("home.html",prediction_text="Similary in text1 and text2 is {output}".format(output=results))
    # return {"similarity score":str(results)}
    

    
    # return render_template("home.html",prediction_text="The House price prediction is {}".format(output))
if __name__=="__main__":
    app.run(debug=True)
