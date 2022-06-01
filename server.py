from distutils.log import debug
from enum import auto
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
import cv2 as cv

app = Flask(__name__)

@app.route('/', methods = ['GET'])
def index():
	title="Halaman Home"
	kontent="Anemia"
	return render_template('index.html',title=title,kontent=kontent)

# Fitur Auto Crop
def auto_crop(crop):
	cropped_image = crop[50:200, 180:200]
	crop_resize=cv.resize(cropped_image,(256,256))
	return crop_resize


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
	x=auto_crop(x)
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
	title="Halaman Prediksi"
	kontent="Anemia"
	return render_template('prediksi.html',title=title,kontent=kontent
	)

@app.route('/prediction', methods = ['GET', 'POST'])
def upload_file():
		if request.method == 'POST':
			file = request.files['file']
			filename = file.filename
			file_path = os.path.join(r'static/prediksi/', filename)                    
			file.save(file_path)
			print(filename)
			product = predict_img(file_path)
			print(product)
			return render_template('prediksi.html',palpeb=product)
		else :
			masukkan_gambar="Masukkan gambar anda terlebih dahulu"
			return render_template('prediksi.html',peringatan=masukkan_gambar)	


    
if __name__ == '__main__':
    app.run(debug=True)