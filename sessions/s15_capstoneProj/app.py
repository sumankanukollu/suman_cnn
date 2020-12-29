from flask import Flask
from pathlib import Path

#UPLOAD_FOLDER = os.path.join(os.getcwd(),'datasets') #'D:/uploads'
UPLOAD_FOLDER = Path.joinpath(Path.cwd(),'datasets')

if not UPLOAD_FOLDER.exists():
    UPLOAD_FOLDER.mkdir()
    print('### Dataset dir is created')
    

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['dspth'] = None
app.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024
app.config['noImgs'] = 100
app.config['imsz'] = (224,224)
app.config['DEBUG'] = True

# Training parameters
app.config['bs'] = 16
app.config['lr'] = 0.1
app.config['epochs'] = 1
app.config['val_ds_per'] = 0.0
app.config['classList'] = []