from distutils.log import debug
import os
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

@app.route('/', methods = ['GET'])
def index():
	return render_template('index.html')

def take_model():
	try:
		global model
		model = load_model('model/imageclassifier.h5')
		print("Load model success !!")
	except Exception as e:
		print(e)

def load_img(img_p):
		
	img = image.load_img(img_p, target_size=(256, 256))
	x = image.img_to_array(img)
	x = np.expand_dims(x,0)
	x=np.vstack([x])
	return x
	
def predict_img(img_):
	new_image = load_img(img_)
	classes = model.predict(new_image, batch_size=10)
	print(classes)
	if classes>=0.9:
		hasil= "Anemia"
	else :
		hasil= "Normal"
	return hasil	 

take_model()
		

@app.route('/prediction', methods = ['GET'])
def prediction():
	return render_template('prediksi.html'
	)

@app.route('/prediction', methods = ['GET', 'POST'])
def upload_file():
		if request.method == 'POST':
			file = request.files['file']
			filename = file.filename
			file_path = os.path.join(r'static/', filename)                       #slashes should be handeled properly
			file.save(file_path)
			print(filename)
			product = predict_img(file_path)
			print(product)
			return render_template('prediksi.html',palpeb=product)


    
if __name__ == '__main__':
    app.run(debug=True)