# https://github.com/roytuts/flask/tree/master/python-flask-upload-display-image
# https://blog.miguelgrinberg.com/post/handling-file-uploads-with-flask

import os,shutil,urllib.request,pdb

from pprint import pprint
from pathlib import Path
import pandas as pd
#from app import app
from flask import Flask, flash, request, redirect, render_template
from PIL import Image
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

import train_sentimentAnalysis
from train_sentimentAnalysis import *

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'csv'])

####################
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
app.config['epochs'] = 3
app.config['val_ds_per'] = 0.0
app.config['classList'] = []


# Saved Datasets
app.config['datasets'] = {}
####################

def getNumberOfClasses(filesList):
    tmp = []
    for f in filesList:
        cls = Path(f.filename).parts[1]
        if cls not in tmp:
            tmp.append(cls)
    return tmp
    
def getSetofImgs(filesList,cls=None):    
    tmp = []
    for f in filesList:
        if Path(f.filename).parts[1] == cls:
            tmp.append(f)
    return tmp[:app.config['noImgs']]
        

def resizeNdSaveImage(im,filepath):
	try:
		im = im.resize(app.config['imsz'])
		im.save(filepath)
	except Exception as e:
		print(str(e))
        
        
# Calculate Mean and Std
def calcMeanStd(dl):
    mean = 0.
    std = 0.
    nb_samples = 0.
    for data,lbl in dl:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples
    mean /= nb_samples
    std /= nb_samples
    #print('Mean is \t: {}\nstd is \t: {}\nnb_samples \t:{}'.format(mean,std,nb_samples))
    return {'mean': mean, 'std' : std, 'noOfImgs' : nb_samples}
    
def startTrainModel(dspth):
    import torch
    global model
    model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=False)
    import torchvision,PIL
    import numpy as np
    
    SEED = 1234
    dspth = dspth
    global ds
    # 1. Load Dataset
    transform = torchvision.transforms.Compose([
                    torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
                    torchvision.transforms.ToTensor()
                ])
    ds = torchvision.datasets.ImageFolder(dspth, transform=transform)
    dl = torch.utils.data.DataLoader(ds, batch_size=256,shuffle=True)
    #print(f'### Dataset classes are : {ds.classes}')
    
    norm = calcMeanStd(dl)
    #print('### Meand and std are : {}'.format(tuple(norm['mean'].numpy()), tuple(norm['std'].numpy())))


    transform = torchvision.transforms.Compose([
                    torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                            mean= tuple(norm['mean'].numpy()), 
                            std = tuple(norm['std'].numpy())
                        )
                ])

    ds = torchvision.datasets.ImageFolder(dspth, transform=transform)
    dl = torch.utils.data.DataLoader(ds, batch_size=256,shuffle=True)
    
    cuda = torch.cuda.is_available()
    #print('### cuda Available or not ? : {}'.format(cuda))
    dataloader_args = dict(shuffle=True, batch_size=app.config['bs'], num_workers=1, pin_memory=True) if cuda else dict(shuffle=True, batch_size = app.config['bs'])

    trainds, testds = torch.utils.data.random_split(dataset=ds, lengths=[len(ds)-int(app.config['val_ds_per']*len(ds)), int(app.config['val_ds_per']*len(ds)) ])
    print('### Length of Train DS is : {} and Length of Test DS is  : {}'.format(len(trainds), len(testds)))
    
    traindl = torch.utils.data.DataLoader(trainds,  **dataloader_args)
    testdl  = torch.utils.data.DataLoader(testds, **dataloader_args)

    print('### Dataset contains {} - classes : {}'.format(len(ds.classes),dl.dataset.class_to_idx))
    #print(f'### Train DL mean and std : {calcMeanStd(traindl)}')
    
    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.classifier[1]

    for param in model.parameters():
        param.requires_grad = False    

    n_inputs = model.classifier[1].in_features
    #print(n_inputs)

    model.classifier = torch.nn.Sequential(
                            torch.nn.Linear(n_inputs, 512),
                            torch.nn.Linear(512, len(app.config['classList'])),
                            torch.nn.LogSoftmax(dim=1)
                        )
    model.to(device)
    # Loss
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr = app.config['lr'])
    #print(model)
    
    def train_model(model, batch_size, n_epochs):
        train_losses,test_losses  = [],[]
        avg_train_losses, avg_test_losses = [],[]
        total, correct, wrong, acc = 0,0,0,0
        misclass_imgList = []
        misclass_imgList = np.array(misclass_imgList)
        for epoch in range(1,n_epochs+1):
            print(f'### Trainig is in progress for epoch : {epoch}')
            ##### Trainnig:
            model.train()
            for inputs, target in traindl:
                inputs, target = inputs.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(inputs)
                loss   = criterion(output,target)
                optimizer.step()
                train_losses.append(loss.item())
            
            ##### Test model:
            model.eval()
            with torch.no_grad():
                for inputs,target in testdl:
                    inputs, target = inputs.to(device), target.to(device)
                    output = model(inputs)
                    _, predicted = torch.max(output.data, 1)
                    loss = criterion(output,target)
                    test_losses.append(loss.item())
                    
                    total   += target.size(0)
                    correct += (predicted == target).sum().item()
                    wrong   += (predicted != target).sum().item()
                    print('### Total : {}, corrct : {} and wrong : {}'.format(total,correct,wrong))
                    misclass_imgList = np.append(misclass_imgList, ((predicted==target)==False).nonzero().view(-1).numpy())
                
            # calculate average loss over an epoch
            train_loss = np.average(train_losses)
            test_loss  = np.average(test_losses)
            avg_train_losses.append(train_loss)
            avg_test_losses.append(test_loss)
            test_acc =  (correct / total) * 100.0
            
            epoch_len = len(str(n_epochs))
            print_msg = (f'\n[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                         f'train_loss: {train_loss:.4f} ' +
                         f'\ttest_loss: {test_loss:.4f}' +
                         f'\tTest_accuracy: {test_acc:.2f}')
            print(print_msg)
            
        return  model, avg_train_losses, avg_test_losses,misclass_imgList,total,correct,wrong, test_acc
                
    model, train_loss, test_loss,misclass_imgList,total,correct,wrong,test_acc = train_model(model, batch_size=app.config['bs'], n_epochs=app.config['epochs'])
    #import pdb;pdb.set_trace()
    global modelpth
    modelpth = Path.joinpath(dspth.parent,dspth.name+'.pt')
    torch.save(model.state_dict(),Path.joinpath(dspth.parent,dspth.name+'.pt'))
    return (model, train_loss, test_loss,misclass_imgList,total,correct,wrong,test_acc)
            
    
    
####################
@app.route('/')
def homepage():
    return render_template('/index.html')
    
@app.route('/classification')
def classification():
    print('### Selected project : classification')
    return render_template('/classify.html', train = 'true', validation='true')
    
@app.route('/sentimentAnalysis')
def sentimentAnalysis():
    print('### Selected project : Sentiment Analysis')
    return render_template('sentiment.html')
    
    
@app.route('/csvupload',methods = ['GET','POST'])
def csvupload():
    print('### CSV upload hit : {}'.format(request.files))
    uploaded_file = request.files['file']
    print(f'### Uploaded file is : {uploaded_file.filename}')
    try:
        if uploaded_file.filename == '':
            flash('Please select a file to upload','error')
            return render_template('sentiment.html')
        if uploaded_file.filename != '':
            if not Path.joinpath(app.config['UPLOAD_FOLDER'], uploaded_file.filename).parent.exists():
                Path.joinpath(app.config['UPLOAD_FOLDER'], uploaded_file.filename).parent.mkdir(exist_ok=True,parents=True)
            file_path = Path.joinpath(app.config['UPLOAD_FOLDER'], request.form['pname']+'_'+uploaded_file.filename)
            # set the file path and save in local and copy in s3 bucket
            uploaded_file.save(file_path)
            train_sentimentAnalysis.upload_to_s3(file_path,file_path.name)
            app.config['datasets'][request.form['pname']] = {'dataset' : file_path.name}
            df = pd.read_csv(file_path).head(20)
        return render_template('sentiment.html',name=request.form['pname']+'_'+uploaded_file.filename,data = df.to_html())
    except Exception as e:
        print(f'### Exception is : {str(e)}')
        flash('Exception occurred : {}'.format(str(e)),'error')
        return render_template('sentiment.html')


@app.route('/train_sa',methods = ['POST'])
def train_sa():
    try:
        if request.method == 'POST':
            fileName = request.form['pname']
            if not Path.joinpath(app.config['UPLOAD_FOLDER'],fileName).exists():
                raise Exception ('File doesn\'t exists to train') #train_sentimentAnalysis.downloadfile_from_s3()
            train_dict = train_sentimentAnalysis.train_model(Path.joinpath(app.config['UPLOAD_FOLDER'],fileName )) 
        return render_template('sentiment.html',train_dict = train_dict)
    except Exception as e:
        print(f'### Exception is : {str(e)}')
        flash('Exception occurred : {}'.format(str(e)),'error')
        return render_template('sentiment.html')
    
    
@app.route('/predict', methods = ["GET","POST"])
def sa_inference():
    if request.method=='POST':
        print('### Inference Sentiment Analysis')
        sentence  =  request.form.get('review')
        print(f'### User entered text is : {sentence}')
        modelName =  "model.pt"
        textFields=  "TEXT_fields.pkl"
        result = train_sentimentAnalysis.predict_sentiment(sentence, modelName, textFields)
        print(f'### Inference Result is : {result}')
        if result.item()>=0.5:
            res = "Possitive"
        else:
            res = "Negative"
        print(f'### Result is : {res}')
    return render_template('sentiment.html', result = res)
    

@app.route('/upload',methods = ['POST'])
def upload():
    global imgs_per_cls
    imgs_per_cls = 20
    try:
        if request.method == 'POST':
            print('### Uploading Dataset for classification problem')
            imgSet = {}
            uploadFilesList = request.files.getlist('files[]')
            app.config['classList'] = getNumberOfClasses(uploadFilesList)
            len_classList = len(app.config['classList'])
            for cls in app.config['classList']:
                imgSet[cls] = getSetofImgs(uploadFilesList,cls=cls)
            for cls in imgSet:
                for file in imgSet[cls][:imgs_per_cls]:
                    if '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() not in ALLOWED_EXTENSIONS :
                        flash('File should be in the specified format : {}, file : {} is not proper'.format(ALLOWED_EXTENSIONS,file.filename))
                        len_classList = 0
                        break
                    
                    if len(Path(file.filename).parts) != 3:
                        flash('Dataset structure should be like : "rootDirectory/subDirectory/Images", --> ***(Not like File : {}) which is not met the requirement'.format(file.filename))
                        len_classList = 0
                        break
                        
                    if len(Path(file.filename).parts) == 3 and '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS :
                        filepath = app.config['UPLOAD_FOLDER'].joinpath(file.filename)
                        app.config['dspth'] = filepath.parent.parent
                        if not filepath.exists():
                            filepath.parent.mkdir(exist_ok=True,parents=True)
                        file.save(filepath)
                        im = Image.open(filepath)
                        if im.size != app.config['imsz']: 
                            resizeNdSaveImage(im,filepath)
            
                    
        return render_template('/classify.html',
                                isUploaded      = "true",
                                len_classList   = len_classList,
                                dspth           = app.config['dspth'],
                                train           = 'true'
                            )
    except Exception as e:
        print(f'### Exception is : {str(e)}')
        flash('Exception occurred in upload dataset : {}'.format(str(e)),'error')
        
        return render_template('/classify.html')
                
@app.route('/clear',methods =['POST'])
def clearDataset():
    if request.method == 'POST':
        try:
            shutil.rmtree(app.config['UPLOAD_FOLDER'])
            print('\n\n\n*********** Deleted the dataset Directory ***********')
        except Exception as e:
            print(str(e))
    #return redirect('/')
    return render_template('index.html')


@app.route('/train',methods=['GET','POST'])
def train():
    try:
        if request.method == 'POST':
            print('### Trainnig is in progress for classification')
            app.config['val_ds_per'] = float(request.form['val_ds_per'])/100
            print('### Validation percentage given by yser : {}'.format(app.config['val_ds_per']))
            (model, train_loss, test_loss,misclass_imgList,total,correct,wrong,test_acc) = startTrainModel(dspth = app.config['dspth'])
            #request.form['<b> Uploaded Path is : </b>'])
        return render_template('/classify.html',
                                len_classList   = len(app.config['classList']),
                                val_ds_per      = app.config['val_ds_per'],
                                isUploaded      = "true",
                                dspth           = app.config['dspth'],
                                total           = total,
                                correct         = correct,
                                wrong           = wrong, 
                                test_acc        = test_acc,
                                train           = "true",
                                validation      = "true",
                                inference       = 'true')
    except Exception as e:
        print(f'### Exception is : {str(e)}')
        flash('Exception occurred in Train, May be flask app restarted. So, please upload dataset set and train again : \n{}'.format(str(e)),'error')
        return render_template('/classify.html')
        
        
        
@app.route('/inference',methods=['GET', 'POST'])
def inference():
    try:
        if request.method == 'POST':
            f = request.files['file']
            testImgPth = app.config['UPLOAD_FOLDER'].joinpath(f.filename)
            f.save(testImgPth)
            chkpnt = torch.load(modelpth)
            model.load_state_dict(chkpnt)
            model.eval()
            im = Image.open(testImgPth)
            if im.size != app.config['imsz']: 
                resizeNdSaveImage(im,testImgPth)
            
            im = Image.open(testImgPth)
            from torchvision import transforms 
            im1 = transforms.ToTensor()(im)
            
            with torch.no_grad():
                 _, cls = torch.max(model(im1.unsqueeze(0)).data,1)
                 
            return render_template('/classify.html',
                                    train           = "true",
                                    predictedcls    = ds.classes[cls],
                                    validation      = "true",
                                    inference       = 'true')
        
    except Exception as e:
        print(f'### Exception is : {str(e)}')
        flash('Exception occurred during Inference, May be flask app restarted. So, please train the dataset set and Test again : \n {}'.format(str(e)),'error')
        return render_template('/classify.html')
        
                                

@app.errorhandler(413)
def request_entity_too_large(error):
    return 'File Too Large', 413

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=80,debug=True)
