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
MODEL_PATH  = os.environ['MODEL_FILE_NAME_KEY'] if 'MODEL_FILE_NAME_KEY' in os.environ else 'netG_chkpt_1840_torch1.5_traced.pth'
print('### S3 Bkt is : {} \nModel path is : {}'.format(S3_BUCKET,MODEL_PATH))

# Create client to AWS S3
s3 = boto3.client('s3') 

def load_model_from_s3bkt():
    try:
        obj         = s3.get_object(Bucket = S3_BUCKET, Key = MODEL_PATH)
        bytestream  = io.BytesIO(obj['Body'].read())
        print('### Loading model...')
        print(f'Model Size: {sys.getsizeof(bytestream) // (1024 * 1024)}')
        model = torch.jit.load(bytestream, map_location=torch.device('cpu'))
        print('### Model is loaded and returning model')
        return model
    except Exception as e:
        print('### Exception in loading a model : {}'.format(str(e)))
        raise(e)

model   = load_model_from_s3bkt()
device  = 'cpu'

def dcGAN_car(event, context):
    try:
        print('### You are in handler dcGAN_car function')
        print('### event is : {}'.format(event))
        print('### Context is : {}'.format(context))
        #content_type_header = event['headers']['content-type']
        #print('### content_type_header : ',content_type_header)
        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_tensor(t):
            norm_ip(t, float(t.min()), float(t.max()))

        with torch.no_grad():
            fake = model(torch.randn(1, 64, 1, 1).to(device)).detach().cpu()
        fakei = fake[0]
        norm_tensor(fakei)
        print('### fakei shape is :',fakei.shape)
        
        # Convert to numpy:
        npImg = fakei.permute(1, 2, 0).numpy()
        pil_img = Image.fromarray((npImg * 255).astype(np.uint8))
        buff = io.BytesIO()

        pil_img.save(buff, format="JPEG")
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