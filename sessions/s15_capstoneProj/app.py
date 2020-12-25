from flask import Flask
import os

UPLOAD_FOLDER = os.path.join(os.getcwd(),'datasets') #'D:/uploads'

if not os.path.exists(UPLOAD_FOLDER):
	os.mkdir(UPLOAD_FOLDER)
	print('### Dataset dir is created')

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024
app.config['imsz'] = (224,224)
app.config['DEBUG'] = True