from flask import Flask,jsonify,request
import tensorflow as tf
import numpy as np
from keras.models import load_model
from PIL import Image
app = Flask(__name__)


model = load_model("potato.h5", compile=False)

class_names=['Potato__Early_blight', 'Potato_Late_blight', 'Potato__healthy']

def predict(model, img_path, class_names):
    img = Image.open(img_path)
    img = img.resize((256, 256))  
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence



@app.route('/')
def hello_world():
   return 'Hello World'

@app.route('/predict', methods=['POST'])
def predict_img():
    if (request.method=="POST"):
        imagefile=request.files['image']
        predicted_class, confidence = predict(model, imagefile, class_names)
        return jsonify({
            "Prediction":predicted_class,
            "confidence":confidence,
            })


if __name__ == '__main__':
    app.run(host='0.0.0.0')