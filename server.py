from flask import *
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
import os

app = Flask(__name__)



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

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/prediksi")
def prediksi():
    return render_template('prediksi.html')
    

@app.route('/prediction', methods = ['GET', 'POST'])
def upload_file():
		if request.method == 'POST':  # Jika user post maka
			file = request.files['file'] # jika user memasukkan file gambar 
			filename = file.filename  
			file_path = os.path.join(r'static/prediksi/', filename) # maka disimpan k folder static/prediksi/ sesuai dengan nama file tersebut                    
			file.save(file_path)  #serta disimpan ke file_path
			print(filename)   #menampilkan nama upload user di konsole
			product = predict_img(file_path)  # Lakukan prediksi upload user yang sudah disimpan
			print(product)   # Hasil prediksi keluar di output untuk di konsole


			# #  Bagaimana cara membuat gambar user yang di auto agar disimpan di static/crop ?
			# file_name_crop=file.filename
			# file_crop=os.path.join(r'static/crop/',file_name_crop)
			# try:
			# 	file.save(file_crop)
			# 	save=auto_crop_user(file_crop)
			# 	file.save(save)
			# except Exception as e :
			# 	print(e)	

				# Untuk Melempar nama ke front end
			title="Halaman Prediksi"
			kontent="Teknologi"
			tim_pertama="Muhammad Reza"
			tim_kedua="Fahmi Yusron Fiddin"
			tim_ketiga="Ni Wayan Erna"
			tim_kempat="Helsa"
			tim_kelima="Elva"	
			return render_template('prediksi.html',palpeb=product,user_img=file_path,title=title,kontent=kontent,
	tim_pertama=tim_pertama,
	tim_kedua=tim_kedua,
	tim_ketiga=tim_ketiga,
	tim_kempat=tim_kempat,
	tim_kelima=tim_kelima) 

if __name__ == "__main__":
    app.run(debug=True)