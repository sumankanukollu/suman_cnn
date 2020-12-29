import os,shutil,urllib.request

from pprint import pprint
from pathlib import Path

from app import app
from flask import Flask, flash, request, redirect, render_template
from PIL import Image
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'csv'])

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
    
def startTrainModel():
    import torch,torchvision,PIL
    import numpy as np
    
    SEED = 1234
    dspth = app.config['dspth']

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
    model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=False)

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
                    #import pdb;pdb.set_trace()
        
                
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
    return (model, train_loss, test_loss,misclass_imgList,total,correct,wrong,test_acc)
            
    

####################
@app.route('/')
def homepage():
    return render_template('/home.html')
    
    
@app.route('/upload',methods = ['POST'])
def upload():
    if request.method == 'POST':
        tmp = {}
        uploadFilesList = request.files.getlist('files[]')
        app.config['classList'] = getNumberOfClasses(uploadFilesList)
        len_classList = len(app.config['classList'])
        for cls in app.config['classList']:
            tmp[cls] = getSetofImgs(uploadFilesList,cls=cls)
        for cls in tmp:
            for file in tmp[cls]:
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
        app.config['val_ds_per'] = float(request.form['val_ds_per'])/100
                
    return render_template('/home.html',len_classList=len_classList,isUploaded="true",dspth = app.config['dspth'])
                
@app.route('/clear',methods =['POST'])
def clearDataset():
    if request.method == 'POST':
        try:
            shutil.rmtree(app.config['UPLOAD_FOLDER'])
            print('\n\n\n*********** Deleted the dataset Directory ***********')
        except Exception as e:
            print(str(e))
    #return render_template('/home.html')
    return redirect('/')


@app.route('/train',methods=['POST'])
def train():
    if request.method == 'POST':
        print('### Train is in progress')
        #import pdb;pdb.set_trace()
        (model, train_loss, test_loss,misclass_imgList,total,correct,wrong,test_acc) = startTrainModel()
    return render_template('/home.html',
                            len_classList   = len(app.config['classList']),
                            val_ds_per      = app.config['val_ds_per'],
                            isUploaded      = "true",
                            dspth           = app.config['dspth'],
                            total           = total,
                            correct         = correct,
                            wrong           = wrong, 
                            test_acc        = test_acc)

    

@app.errorhandler(413)
def request_entity_too_large(error):
    return 'File Too Large', 413

if __name__ == "__main__":
    app.run(debug=True)