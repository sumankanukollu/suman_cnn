try:
    print("### Import START...")
    import unzip_requirements
    from requests_toolbelt.multipart import decoder

    import torch,base64,boto3,os,io,json,sys

    import torch.nn as nn
    import numpy as np
    from PIL import Image
    from io import BytesIO

    print('### Using Torch version :',torch.__version__)
except Exception as e:
    print('### Exception occured while importing modules : {}'.format(str(e)))

# define env variables if there are not existing
S3_BUCKET   = os.environ['MODEL_BUCKET_NAME'] if 'MODEL_BUCKET_NAME' in os.environ else 'suman-p2-bucket'
MODEL_PATH  = os.environ['MODEL_FILE_NAME_KEY'] if 'MODEL_FILE_NAME_KEY' in os.environ else 's7vae_bestvloss_chkpt_0324.pth'
print('### S3 Bkt is : {} \nModel path is : {}'.format(S3_BUCKET,MODEL_PATH))


# define a simple linear VAE
class LinearVAE(nn.Module):
    def __init__(self):
        super(LinearVAE, self).__init__()
        # Encoder 
        self.encoder = nn. Sequential(
            nn.Linear(in_features=3*64*64, out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=32)
        )

        # Decoder 
        self.decoder = nn. Sequential(
            nn.Linear(in_features=32, out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=3*64*64)
        )

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample
 
    def forward(self, x):
        # encoding
        x = self.encoder(x)

        # get `mu` and `log_var`
        mu = x
        log_var = x
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)

        # decoding
        x = self.decoder(z)
        reconstruction = torch.sigmoid(x)
        return reconstruction, mu, log_var


# Create client to AWS S3
s3 = boto3.client('s3') 



def load_model_from_s3bkt():
    try:
        obj         = s3.get_object(Bucket = S3_BUCKET, Key = MODEL_PATH)
        bytestream  = io.BytesIO(obj['Body'].read())
        print('### Loading model...')
        print(f'Model Size: {sys.getsizeof(bytestream) // (1024 * 1024)} MB')
        #model = torch.jit.load(bytestream, map_location=torch.device('cpu'))
        model = LinearVAE().to('cpu')
        chkpoint = torch.load(bytestream,map_location=torch.device('cpu')) 
        model.load_state_dict(chkpoint['model_statedict'])
        print(model)
        print('### Model is loaded and returning model')
        return model
    except Exception as e:
        print('### Exception in loading a model : {}'.format(str(e)))
        raise(e)

model   = load_model_from_s3bkt()
model.eval()
device  = 'cpu'

def s7VariationalAutoencoders(event, context):
    try:
        print('### You are in handler s7VariationalAutoencoders function')
        print('### event is : {}'.format(event))
        print('### Context is : {}'.format(context))
        with torch.no_grad():
            sample = torch.randn(32, 32).to(device)
            sample = model.decoder(sample).cpu()
            sample = sample.view(-1, 3, 64, 64)[:1]
        
        print(f'### Sample Image shape is : {sample.shape}')
        # Convert to numpy:
        im_np = sample[0].permute(1, 2, 0).numpy()
        print(f'### Numpy Image shape is : {im_np.shape}')
        pil_img = Image.fromarray((im_np).astype(np.uint8))
        print(f'### pil image size is : {pil_img.size}')
        buff = io.BytesIO()
        pil_img.save(buff, format="JPEG")
        print(f'### Contents of the buff : {buff.getvalue()}')
        new_image_string = base64.b64encode(buff.getvalue()).decode("utf-8")
        img_str = f"data:image/jpeg;base64,{new_image_string}"
        
        print('### Final Image String is : \n\t{}'.format(img_str))

        return {
            "statusCode": 200,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True

            },
            "body": json.dumps({'imagebytes': img_str})
        }
    except Exception as e:
        print(repr(e))
        return {
            "statusCode": 500,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({"error": repr(e)})
        }