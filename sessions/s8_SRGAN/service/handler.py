try:
    print("### Import START...")
    import unzip_requirements
    from requests_toolbelt.multipart import decoder
    
    import torch,base64,boto3,os,io,json,sys,math
    import torch.nn as nn
    import numpy as np
    import torchvision
    from torchvision import transforms
    from torchvision.utils import save_image

    from PIL import Image
    from io import BytesIO

except Exception as e:
    print('### Exception occured while importing modules : {}'.format(str(e)))

# define env variables if there are not existing
S3_BUCKET   = os.environ['MODEL_BUCKET_NAME'] if 'MODEL_BUCKET_NAME' in os.environ else 'suman-p2-bucket'
MODEL_PATH  = os.environ['MODEL_FILE_NAME_KEY'] if 'MODEL_FILE_NAME_KEY' in os.environ else 's8_SRGAN_netG_epoch_2_50.pth'
device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('### S3 Bkt is : {} \nModel path is : {}'.format(S3_BUCKET,MODEL_PATH))

# Create client to AWS S3
s3 = boto3.client('s3') 

class Generator(nn.Module):
    def __init__(self, scale_factor):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (torch.tanh(block8) + 1) / 2


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x

def load_model_from_s3bkt():
    try:
        obj         = s3.get_object(Bucket = S3_BUCKET, Key = MODEL_PATH)
        print('### Loading model...')
        bytestream  = io.BytesIO(obj['Body'].read())
        print(f'Model Size: {sys.getsizeof(bytestream) // (1024 * 1024)} MB')
        scale_factor = 2
        netG = Generator(scale_factor=scale_factor).to(device)
        netG.load_state_dict(
            torch.load(bytestream,
                       map_location=torch.device('cpu')
            )
        )
        netG.eval()
        return netG
    except Exception as e:
        print('Exception in loading a model')
        print(repr(e))
        raise(e)

netG = load_model_from_s3bkt()

def convertImage2Tensor(image_bytes):
    print('### Transformation Start...')
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img_t = transforms.ToTensor()(img)
        return img_t
    except Exception as e:
        print('### Exception in transforming an image')
        print(str(e))
        raise(e)

def get_SRGAN_image(image_bytes):
    print('Getting Prediction...')
    img_t = convertImage2Tensor(image_bytes=image_bytes)
    with torch.no_grad():
        srimg = netG(img_t.unsqueeze(0)).detach().cpu()
        srimg = srimg[0]
    return srimg


def SRGAN(event, context):
    try:
        print('### You are in handler SR-GAN function')
        print('### event is   : {}'.format(event))
        print('### Context is : {}'.format(context))
        
        content_type_header = event['headers']['content-type']
        body    = base64.b64decode(event["body"])
        print(f'### Body is : {body}')
        picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
        
        image_data = picture.content
        img = Image.open(io.BytesIO(image_data))
        print(f'### Type of img : {type(img)}')
        
        img_t = transforms.ToTensor()(img)

        with torch.no_grad():
            srimg = netG(img_t.unsqueeze(0)).detach().cpu()
            srimg = srimg[0]
            
        srimg_np = srimg.permute(1,2,0).numpy()
        
        pil_img = Image.fromarray((srimg_np * 255).astype(np.uint8))
        #pil_img = Image.fromarray((srimg_np).astype(np.uint8))
        buff = BytesIO()
        print(f'### pil image size is : {pil_img.size}')

        print(f'### buff is : {buff.getvalue()}')
        pil_img.save(buff, format="JPEG")
        new_image_string = base64.b64encode(buff.getvalue()).decode("utf-8")
        img_str = f"data:image/jpeg;base64,{new_image_string}"
        print(f'### Final Image String is : {img_str}')

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