import flask
from flask import Flask,render_template,url_for,request
import pickle
import base64
import numpy as np
import cv2
#import tensorflow as tf
import torch
from PIL import Image

#import pdb;pdb.set_trace()
#Initialize the useless part of the base64 encoded image.
init_Base64 = 21;

#Our dictionary
label_dict = dict(zip(range(0,10),range(0,10)))
print(f'### Label dict : {label_dict}')

#Initializing the Default Graph (prevent errors)
#graph = tf.get_default_graph()

# Use pickle to load in the pre-trained model.
#with open(f'model_cnn.pkl', 'rb') as f:
#        model = pickle.load(f)
device = 'cpu' if not torch.cuda.is_available() else 'cuda'
print(f'### device : {device}')

# Model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        dout = nn.Dropout(0.1)
        self.conv1 = nn.Sequential(
          nn.Conv2d(1, 16, 3),
          nn.ReLU(),
          nn.BatchNorm2d(16),
          dout
        ) #26
        self.conv2 = nn.Sequential(
          nn.Conv2d(16, 32, 3),
          nn.ReLU(),
          nn.BatchNorm2d(32),
          dout
        ) #24
        
        self.conv3 = nn.Conv2d(32,10,1) #24

        self.pool1 = nn.MaxPool2d(2,2) #12

        self.conv4 = nn.Sequential(
          nn.Conv2d(10, 10, 3),
          nn.ReLU(),
          nn.BatchNorm2d(10),
          dout
        ) #10
        
        self.conv5 = nn.Sequential(
          nn.Conv2d(10, 16, 3),
          nn.ReLU(),
          nn.BatchNorm2d(16),
          dout
        ) #8
        
        self.conv6 = nn.Sequential(
          nn.Conv2d(16, 16, 3),
          nn.ReLU(),
          nn.BatchNorm2d(16),
          dout
        ) #6
        
        self.conv7 = nn.Sequential(
          nn.Conv2d(16, 16, 3),
          nn.ReLU(),
          #nn.BatchNorm2d(16),
          dout
        ) #4
        
        self.conv8 = nn.Sequential(
          nn.Conv2d(16, 10, 4),
          #nn.ReLU(),
          #nn.BatchNorm2d(10),
          dout          
        ) #1
        
    def forward(self, x):
        x = self.conv1(x)  # i = 28  o = 26  RF= 3
        x = self.conv2(x)  # i = 26  o = 24  RF= 5
        x = self.conv3(x)  # i = 24  o = 24  RF= 5
        x = self.pool1(x)  # i = 24  o = 12  RF= 10
        x = self.conv4(x)  # i = 12  o = 10  RF= 12
        x = self.conv5(x)  # i = 10  o = 8   RF= 14
        #x = self.pool1(x)
        x = self.conv6(x)  # i = 8  o = 6  RF= 16
        x = self.conv7(x)  # i = 6  o = 4  RF= 18
        x = self.conv8(x)  # i = 4  o = 1  RF= 20
        #set_trace()
        x = x.view(-1, 10)
        #return F.log_softmax(x)
        #nn.Flatten(x)
        return F.softmax(x)

        
model = Net().to(device)
chkpnt = torch.load('mnist_v3.pt',map_location=torch.device('cpu'))
model.load_state_dict(chkpnt)
model.eval()

#Initializing new Flask instance. Find the html template in "templates".
app = flask.Flask(__name__, template_folder='templates')

#First route : Render the initial drawing template
@app.route('/')
def home():
	return render_template('draw.html')

#Second route : Use our model to make prediction - render the results page.
@app.route('/predict', methods=['POST'])
def predict():
    #global graph
    #with graph.as_default():
    with torch.no_grad():
        if request.method == 'POST':
            final_pred = None
            #Preprocess the image : set the image to 28x28 shape
            #Access the image
            draw = request.form['url']
            #Removing the useless part of the url.
            draw = draw[init_Base64:]
            #Decoding
            draw_decoded = base64.b64decode(draw)
            image = np.asarray(bytearray(draw_decoded), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE) #cv2.IMREAD_COLOR) 
            im = Image.fromarray(image)
            im.save("ActualImg.jpeg")
            #Resizing and reshaping to keep the ratio.
            resized = cv2.resize(image, (28,28) , interpolation = cv2.INTER_AREA)
            im = Image.fromarray(resized)
            im.save("resized.jpeg")
            print(f'### Shape of resized : {resized.shape}')
            vect = np.asarray(resized, dtype="uint8")
            im = Image.fromarray(vect)
            im.save("vect.jpeg")
            vect = vect.reshape(1, 1, 28, 28).astype('float32')
            print(f'### Vect type is : {type(vect)} shape is : {vect.shape} and permuted : {vect.shape}')
            img_t  = torch.from_numpy(vect).float() #tensor
            img_t = img_t.to(device)
            #We do the prediction here and we do + 1 because we start from 0
            my_prediction = model(img_t).detach().numpy()[0].argmax() 
            
            #Launch prediction
            #my_prediction = model.predict(vect)
            print(f'### My prediction : {my_prediction}')
            #Getting the index of the maximum prediction
            #index = np.argmax(my_prediction[0])
            #Associating the index and its value within the dictionnary
            final_pred = label_dict[my_prediction]

    return render_template('results.html', prediction =final_pred)


if __name__ == '__main__':
	app.run(debug=True)
