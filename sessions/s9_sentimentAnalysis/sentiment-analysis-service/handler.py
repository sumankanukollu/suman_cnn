try:
    import unzip_requirements
    import json,boto3,os,tarfile,io,base64,json,pickle
    from requests_toolbelt.multipart import decoder

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchtext import data

    import spacy
    import dill
    import random
    import numpy as np

except ImportError:
    print('### Exception occured in import')

print('### Import End....')

##############
device = "cpu"
S3_BUCKET = os.environ['S3_BUCKET'] if 'S3_BUCKET' in os.environ else 'suman-p2-bucket'
MODEL_PATH = os.environ['MODEL_PATH'] if 'MODEL_PATH' in os.environ else 'sentiment_analysis_model_cnn2_st_dct.pt'
TORCH_TEXT_FIELD =  os.environ['TORCH_TEXT_FIELD'] if 'TORCH_TEXT_FIELD' in os.environ else 'TEXT_fields_cnn2.pkl'

s3 = boto3.client('s3')
##############

##############
#Load Model file:
##############

if os.path.isfile(MODEL_PATH) != True:
    obj = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_PATH)
    print("### Main: Creating model Bytestream...")
    bytestream_model = io.BytesIO(obj['Body'].read())
    print("### Main: Loading Model...")
    model_ckpnt = torch.load(bytestream_model,map_location=torch.device('cpu'))
    print("Main: Model Loaded...")

##############
#Load TEXT vocab file:
##############
if os.path.isfile(TORCH_TEXT_FIELD) != True:
    obj = s3.get_object(Bucket=S3_BUCKET, Key=TORCH_TEXT_FIELD)
    print("### Main: Creating torchtext Bytestream...")
    bytestream_text = io.BytesIO(obj['Body'].read())
    print('### Main: loading torchtext Fields')
    TEXT = torch.load(bytestream_text, pickle_module=dill)
    print("Main: torchtext Fields Model Loaded...")
    
    
##############
#Define Model:
##############
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

#TEXT = data.Field(tokenize = 'spacy', batch_first = True)
LABEL = data.LabelField(dtype = torch.float)

class CNN2(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
                 dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fs, embedding_dim)) 
                                    for fs in filter_sizes
                                    ])
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        #text = [batch size, sent len]
        embedded = self.embedding(text)
        #embedded = [batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1)
        #embedded = [batch size, 1, sent len, emb dim]
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        #pooled_n = [batch size, n_filters]
        cat = self.dropout(torch.cat(pooled, dim = 1))
        #cat = [batch size, n_filters * len(filter_sizes)]
        return self.fc(cat)

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
N_FILTERS = 100
FILTER_SIZES = [3,4,5]
OUTPUT_DIM = 1
DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = CNN2(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'### Device is : {device}')
model = model.to(device)

model.load_state_dict(model_ckpnt) 
model.eval()

print('### loading spacy')
nlp = spacy.load('/tmp/pkgs-from-layer/requirements/en_core_web_sm/en_core_web_sm-2.2.5')



def predict_sentiment(model, sentence, min_len = 5):
    model.eval()
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    if len(tokenized) < min_len:
        tokenized += ['<pad>'] * (min_len - len(tokenized))
    #indexed = [data.stoi[t] for t in tokenized]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(0)
    prediction = torch.sigmoid(model(tensor))
    return prediction.item()
    
    
    
    
def predictSentiment(event, context):
    print(event['body'])
    bodyTemp = event["body"]
    print("Body Loaded")
    body = json.loads(bodyTemp)
    print(body,type(body))
    writtenMovieReview = body["writtenMovieReview"]
    print(writtenMovieReview)
    print(type(writtenMovieReview))
    #model = RNN()
    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.5
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

    predValue = predict_sentiment(model,writtenMovieReview)
    print('### Predicted value',predValue)
    review = "Positive" if predValue >= 0.5 else "Negative"
    
    response = {
        "statusCode": 200,
    #    "body":json.dumps(body)
        "body": json.dumps({"input": writtenMovieReview , "predictionValue":predValue, "sentiment": review})
    }

    return response
