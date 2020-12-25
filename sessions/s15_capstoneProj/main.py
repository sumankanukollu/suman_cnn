import os,shutil,urllib.request

from app import app
from flask import Flask, flash, request, redirect, render_template
from PIL import Image
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

formvalues = {}

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
def resizeImage(im,filename):
	try:
		im = im.resize(app.config['imsz'])
		im.save(os.path.join(formvalues['dataset_cls_dir'],filename))
	except Exception as e:
		print(str(e))
        


# Calculate Mean and Std
def calcMeanStd(dl):
    mean = 0.
    std = 0.
    nb_samples = 0.
    for data,lbl in dl:
        #set_trace()
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples
    mean /= nb_samples
    std /= nb_samples
    print('Mean is \t: {}\nstd is \t: {}\nnb_samples \t:{}'.format(mean,std,nb_samples))
    return {'mean': mean, 'std' : std, 'noOfImgs' : nb_samples}

def startTrainningModel(formvalues):
    import torch,torchvision,os,PIL,torchsummary
    import numpy as np
    SEED = 1234
    bs = 16
    lr = 0.1
    epochs = 1
    print(f'### formvalues : {formvalues}')
    print('### You are in train method')
    #import pdb;pdb.set_trace()
    noOfClasses = formvalues['no_classes']
    val_ds_per = formvalues['val_ds_per']
    print(f'### noOfClasses : {noOfClasses} and val_ds_per : {val_ds_per}')

    dspth = os.path.dirname(formvalues['dataset_cls_dir']) # os.path.join('datasets','E4P2_capstone')
    # Load dataset
    transform = torchvision.transforms.Compose([
            torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
            torchvision.transforms.ToTensor()
            #,torchvision.transforms.Normalize(mean=norm['mean'], std=(0.1747, 0.1697, 0.1862))
    ])

    ds = torchvision.datasets.ImageFolder(dspth, transform=transform)
    dl = torch.utils.data.DataLoader(ds, batch_size=256,shuffle=True)
    print(f'### Dataset classes are : {ds.classes}')
    

    norm = calcMeanStd(dl)
    print('### Meand and std are : {}'.format(tuple(norm['mean'].numpy()), tuple(norm['std'].numpy())))


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
    print('### cuda Available or not ? : {}'.format(cuda))
    dataloader_args = dict(shuffle=True, batch_size=bs, num_workers=1, pin_memory=True) if cuda else dict(shuffle=True, batch_size=bs)

    trainds, testds = torch.utils.data.random_split(dataset=ds, lengths=[len(ds)-int(val_ds_per*len(ds)), int(val_ds_per*len(ds)) ])
    print('### Length of Train DS is : {}\nLength of Test DS is  : {}'.format(len(trainds), len(testds)))
        
    traindl = torch.utils.data.DataLoader(trainds,  **dataloader_args)
    testdl  = torch.utils.data.DataLoader(testds, **dataloader_args)

    print('Dataset contains {} - classes : \n\t{}'.format(len(ds.classes),dl.dataset.class_to_idx))
    print(f'### Train DL mean and std : {calcMeanStd(traindl)}')

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=False)

    model.classifier[1]

    for param in model.parameters():
        param.requires_grad = False    

    n_inputs = model.classifier[1].in_features
    print(n_inputs)

    model.classifier = torch.nn.Sequential(torch.nn.Linear(n_inputs, 512),
                                #torch.nn.ReLU6(inplace=True),
                                # torch.nn.Dropout(0.2),
                                torch.nn.Linear(512, noOfClasses),
                                torch.nn.LogSoftmax(dim=1))
    model.to(device)


    print(torchsummary.summary(model,(3,224,224)))
    
    # Loss
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr = lr)
    # Train and Validate
    def train_model(model, batch_size, patience, n_epochs):
        train_losses,valid_losses, avg_train_losses,avg_valid_losses   = [],[],[],[]
        for epoch in range(1, n_epochs + 1):
            print(f'### Trainig is in progress for epoch : {epoch}')
            ###################
            # train the model #
            ###################
            model.train() # prep model for training
            for inputs, target in traindl:
                inputs, target = inputs.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(inputs)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            
            ######################    
            # Test the model #
            ######################
            model.eval() # prep model for evaluation
            correct = 0
            total = 0
            accuracy = 0
            for inputs, target in testdl:
                inputs, target = inputs.to(device), target.to(device)
                output = model(inputs)
                _, predicted = torch.max(output.data, 1)

                loss = criterion(output, target)
                # record validation loss
                valid_losses.append(loss.item())
                
                total += target.size(0)
                correct += (predicted == target).sum().item()
            
            # print training/validation statistics 
            # calculate average loss over an epoch
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)

            valid_accuracy = (100.0 * correct) / total

            epoch_len = len(str(n_epochs))

            print_msg = (f'\n[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                         f'train_loss: {train_loss:.5f} ' +
                         f'\tvalid_loss: {valid_loss:.5f}' +
                         f'\tvalid_accuracy: {valid_accuracy:.2f}')
            
            print(print_msg)

           
            # clear lists to track next epoch
            train_losses = []
            valid_losses = []
            
            valid_accuracy = round(valid_accuracy, 2)

        return  model, avg_train_losses, avg_valid_losses

    model, train_loss, valid_loss = train_model(model, batch_size=bs, patience=None, n_epochs=epochs)
    
    ###############
    torch.save(model.state_dict(),'capstone_proj.pth')
    formvalues['train_loss'],formvalues['valid_loss'] = train_loss,valid_loss
    return formvalues
    
    
    

def loadModel():
    chkpnt = torch.load('capstone_proj.pth')
    model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=False)
    for param in model.parameters():
        param.requires_grad = False    

    n_inputs = model.classifier[1].in_features
    print(n_inputs)

    model.classifier = torch.nn.Sequential(torch.nn.Linear(n_inputs, 512),
                                #torch.nn.ReLU6(inplace=True),
                                # torch.nn.Dropout(0.2),
                                torch.nn.Linear(512, noOfClasses),
                                torch.nn.LogSoftmax(dim=1))
    model.load_state_dict(chkpnt)
    model.eval()


##########################
@app.route('/')
def upload_form():
	return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        formvalues['no_classes'] = int(request.form['nocls'])
        formvalues['val_ds_per'] = float(request.form['val_ds_per'])/100
        
        print('\n\n### Number of classes : {} and validation DS per : {}'.format(formvalues['no_classes'],formvalues['val_ds_per'])) 
        
        formvalues['dataset_cls_dir'] = os.path.join(app.config['UPLOAD_FOLDER'], request.form['pname'], request.form['clsname']) if request.form['clsname'] else os.path.join(app.config['UPLOAD_FOLDER'],request.form['pname'])
        if not os.path.exists(formvalues['dataset_cls_dir']):
            os.makedirs(formvalues['dataset_cls_dir'])
            print('### Dataset dir is created')
        # check if the post request has the files part
        if 'files[]' not in request.files:
            flash('No file part')
            return redirect(request.url)
        files = request.files.getlist('files[]')
        for file in files[:100]:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(formvalues['dataset_cls_dir'], filename))
                im = Image.open(os.path.join(formvalues['dataset_cls_dir'],filename))
                im = im.convert('RGB') if im.mode != 'RGB' else im
                if im.size != app.config['imsz']: 
                    resizeImage(im,filename)
        flash('100 File(s) resized to {} and uploaded successfully'.format(app.config['imsz']))
    return redirect('/')
		
@app.route('/clear', methods=['POST'])
def clearDataset():
    if request.method == 'POST':
        try:
            shutil.rmtree(app.config['UPLOAD_FOLDER'])
            print('\n\n\n*********** Deleted the dataset Directory ***********')
            #formvalues = {}
        except Exception as e:
            print(str(e))
            #formvalues = {}
    return redirect('/')
    
    
@app.route('/train', methods=['POST'])
def train():
    if request.method == 'POST':
        try:
            print(f'### Formvalues are : {formvalues}')
            data = startTrainningModel(formvalues)
            print(f'### formvalues after training : {data}')
            
        except Exception as e:
            print(str(e))
    #return redirect('/')
    return render_template('upload.html',loss = data['valid_loss'])
	
		

@app.errorhandler(413)
def request_entity_too_large(error):
    return 'File Too Large', 413

if __name__ == "__main__":
    app.run(debug=True)