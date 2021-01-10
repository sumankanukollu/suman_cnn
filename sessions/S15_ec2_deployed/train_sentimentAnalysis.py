import urllib,time,random,pdb
import boto3,torch,spacy,dill, nltk
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from torchtext import data

nltk.download('stopwords')
from nltk.corpus import stopwords

SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

TEXT    = data.Field(tokenize='spacy', stop_words=stopwords.words('english'))
LABEL   = data.LabelField(dtype=torch.float)


EMBEDDING_DIM   = 100
HIDDEN_DIM      = 512
OUTPUT_DIM      = 1
BATCH_SIZE      = 64
N_EPOCHS        = 5
MODEL_FILE      = "model.pt"
BUCKET_NAME     = "suman-p2-bucket"
TEXT_FIELDS_FILE= "TEXT_fields.pkl"

######
REGION_NAME = 'ap-south-1'

session = boto3.Session(
                aws_access_key_id       =   'AKIA5T5ONA2DILKL4XFJ',
                aws_secret_access_key   =   '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',
            )
s3      = session.resource('s3')
#########
#model = NONE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
best_valid_loss = float('inf')
#s3 = boto3.resource(u's3')

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, text):
        #text = [sent len, batch size]
        embedded = self.embedding(text)
        output, hidden = self.rnn(embedded)
        #output = [sent len, batch size, hid dim]
        #hidden = [1, batch size, hid dim]
        assert torch.equal(output[-1,:,:], hidden.squeeze(0))
        return self.fc(hidden.squeeze(0))

def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum()/len(correct)
    return acc

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():    
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)            
            loss = criterion(predictions, batch.label)            
            acc = binary_accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def run_epochs(model, train_iterator, valid_iterator, optimizer, criterion, best_valid_loss):
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), MODEL_FILE)
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
        return train_loss, train_acc, valid_loss, valid_acc

def compute_test_loss_accuracy(model, test_iterator, criterion):
    model.load_state_dict(torch.load(MODEL_FILE))
    test_loss, test_acc = evaluate(model, test_iterator, criterion)
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
    return test_loss, test_acc

def load_data(filePath):
    print(filePath)
    #download_from_s3(filePath, filePath)
    #print(f'{filePath} downloaded from S3')
    fields      = [('text', TEXT), ('label', LABEL)]
    train_data  = data.TabularDataset.splits(path = '',
                                            train = filePath,
                                            format = 'csv',
                                            fields = fields,
                                            skip_header = True
                                        )
    train_data = train_data[0]
    train_data, test_data = train_data.split(split_ratio=0.7, random_state=random.seed(5))
    train_data, valid_data = train_data.split(split_ratio=0.7,random_state=random.seed(2))
    return train_data, test_data, valid_data

def download_from_s3(local_file, remote_file):
    import pdb;pdb.set_trace()
    if not Path.joinpath(Path.cwd(),'model.pt').exists():
        #s3.Bucket(BUCKET_NAME).download_file(remote_file, local_file)
        s3.Bucket(BUCKET_NAME).download_file(str(remote_file), str(local_file))
    return local_file
    
def downloadfile_from_s3(local_file, remote_file):
    import pdb;pdb.set_trace()
    s3.Bucket(BUCKET_NAME).download_file(remote_file, local_file)

def upload_to_s3(localFile, remoteFile):
    s3.Bucket(BUCKET_NAME).upload_file(str(localFile), str(remoteFile))
    uploadedFileUrl = "https://s3-%s.amazonaws.com/%s/%s" % (
        "ap-south-1",
        BUCKET_NAME,
        urllib.parse.quote(remoteFile, safe="~()*!.'"),
    )
    print('S3 uploaded file Url is ' + remoteFile)
    print("S3 url is " + uploadedFileUrl)
    return uploadedFileUrl

def upload_data(filename):
    return upload_to_s3(filename, filename)

def train_model(data_file):
    # Load Data
    train_data, test_data, valid_data = load_data(data_file)
    print(vars(train_data.examples[0]))
    # Build Vocab
    TEXT.build_vocab(train_data, max_size=10000)
    LABEL.build_vocab(train_data)
    INPUT_DIM = len(TEXT.vocab)
    model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    model = model.to(device)
    criterion = criterion.to(device)
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=BATCH_SIZE,
        sort_key=lambda x:len(x.text),
        sort_within_batch=False,
        device=device)
    run_epochs(model, train_iterator, valid_iterator, optimizer, criterion, best_valid_loss)
    test_loss, test_acc = compute_test_loss_accuracy(model, test_iterator, criterion)
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
    uploadedFileUrl = upload_to_s3(MODEL_FILE, MODEL_FILE)
    # Save fields
    print(f'### INPUT DIM is : {INPUT_DIM}')
    torch.save(TEXT, TEXT_FIELDS_FILE, pickle_module=dill)
    upload_to_s3(TEXT_FIELDS_FILE, TEXT_FIELDS_FILE)

    return {
            "test_loss" : test_loss,
            "test_acc" : test_acc,
            "model" : MODEL_FILE,
            "model_url" : uploadedFileUrl,
            "text_fields_file": TEXT_FIELDS_FILE
        }

def predict_sentiment(sentence, modelName, textFields):
    modelLocalFile      = download_from_s3(modelName, modelName)
    textFieldsLocalFile = download_from_s3(textFields, textFields)
    TEXT = torch.load(textFieldsLocalFile, pickle_module=dill)
    
    INPUT_DIM = len(TEXT.vocab)
    model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
    model.load_state_dict(torch.load(modelLocalFile))

    if not Path.joinpath(Path.cwd(),'en_core_web_sm-2.2.5.zip').exists():
        s3.Bucket(BUCKET_NAME).download_file('en_core_web_sm-2.2.5.zip','en_core_web_sm-2.2.5.zip')
        from zipfile import ZipFile
        with ZipFile('en_core_web_sm-2.2.5.zip','r') as z:
            z.extractall()
        
    nlp = spacy.load('en_core_web_sm-2.2.5/en_core_web_sm/en_core_web_sm-2.2.5')
    model.eval()
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    min_len = 5
    if len(tokenized) < min_len:
        tokenized += ['<pad>'] * (min_len - len(tokenized))
    #indexed = [data.stoi[t] for t in tokenized]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(0)
    prediction = torch.sigmoid(model(tensor))
    print(prediction[0])
    print(type(prediction))
    return prediction.max()