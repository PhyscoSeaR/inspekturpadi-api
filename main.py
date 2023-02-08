import flask
import io
import string
import time
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, jsonify, request 
from flask_cors import CORS, cross_origin


model = tf.keras.models.load_model('model2.h5')

def prepare_image(img):
    img = Image.open(io.BytesIO(img))
    img = img.resize((255, 255))
    img = np.array(img)
    img = np.expand_dims(img, 0)
    return img

def predict_result(img):
    predict_class = np.argmax(model.predict(img))

    if predict_class == 0:
        return 'Bacterial'
    elif predict_class == 1:
        return 'Blast' 
    else :
        return 'Brown' 

def predict_percentage(img):
    predict_proba = sorted(model.predict(img)[0])[2]
    return round(predict_proba*100,2),

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/predict', methods=['POST'])
@cross_origin()
def infer_image():
    # Catch the image file from a POST request
    if 'file' not in request.files:
        return "Please try again. The Image doesn't exist"
    
    file = request.files.get('file')

    if not file:
        return

    # Read the image
    img_bytes = file.read()

    # Prepare the image
    img = prepare_image(img_bytes)

    # Return on a JSON format
    return jsonify(prediction=predict_result(img), presentage=predict_percentage(img))
    

@app.route('/', methods=['GET'])
@cross_origin()
def index():
    return 'Machine Learning Inference'

    
if __name__ == '__main__':
    app.run(port=int(os.environ.get("PORT", 8080)),host='0.0.0.0',debug=True)