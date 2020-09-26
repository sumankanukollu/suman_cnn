print("### Import START...")
try:
    import unzip_requirements
except ImportError:
    pass
from requests_toolbelt.multipart import decoder

import os
import io
import base64
import json
import re

import numpy as np
import PIL
from PIL import Image
#from matplotlib import pyplot as plt

from numpy import asarray
print("### Import End...")
################################
MODEL_PATH = os.environ['MODEL_FILE_NAME_KEY'] if 'MODEL_FILE_NAME_KEY' in os.environ else 'simple_pose_estimation_quantized.onnx'
print(MODEL_PATH)
exists = os.path.exists(MODEL_PATH)
print(f'File {MODEL_PATH} exists : {exists}')

def load_model():
    print("Loading model...")
    model_file = MODEL_PATH
    ort_session = onnxruntime.InferenceSession(model_file)
    if ort_session is None:
        print('Failed to load the model')
    print('Model Loadede')
    return ort_session

ort_session = load_model()

def transform_image(image_bytes):
    print('Transformation Start...')
    try:
        transformations = transforms.Compose([
            transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        image = Image.open(io.BytesIO(image_bytes))
        print('Transformation finished...')
        return transformations(image).unsqueeze(0)
    except Exception as e:
        print('Exception in transforming an image')
        print(repr(e))
        raise(e)

def img_to_base64(img):
    img = Image.fromarray(img, 'RGB') 
    buffer = io.BytesIO()
    img.save(buffer,format="JPEG")
    myimage = buffer.getvalue()                     
    img_str = f"data:image/jpeg;base64,{base64.b64encode(myimage).decode()}"
    return img_str

def HWC_2_CHW(img):
    H, W, C = img.shape 
    im = np.zeros((C,H,W),dtype=np.float32)
    for i in range(C):
      im[i] = img[:,:,i]      
    return im

def img_normalize(img, means, stds):
    """
    Args:
        Numpy : Image of size (C, H, W) to be normalized.
    Returns:
        Numpy: Normalized image.
    """
    for i in range(3):
        img[i] = np.divide(np.subtract(img[i], means[i]), stds[i])
        img[i] = np.nan_to_num(img[i])
    return img

def get_hpe_image(image_bytes):
    print('### HPE process started...')
    #### Variables:
    POSE_PAIRS = [
        # UPPER BODY
                      [9, 8],
                      [8, 7],
                      [7, 6],

        # LOWER BODY
                      [6, 2],
                      [2, 1],
                      [1, 0],

                      [6, 3],
                      [3, 4],
                      [4, 5],

        # ARMS
                      [7, 12],
                      [12, 11],
                      [11, 10],

                      [7, 13],
                      [13, 14],
                      [14, 15]
        ]

    JOINTS = ['0 - r ankle', '1 - r knee', '2 - r hip', '3 - l hip', '4 - l knee', '5 - l ankle', '6 - pelvis', '7 - thorax', '8 - upper neck', '9 - head top', '10 - r wrist', '11 - r elbow', '12 - r shoulder', '13 - l shoulder', '14 - l elbow', '15 - l wrist']
    JOINTS = [re.sub(r'[0-9]+|-', '', joint).strip().replace(' ', '-') for joint in JOINTS]
    
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize((256,256),  PIL.Image.BICUBIC) # resizing image
    image = np.asarray(image)
    img_chw = HWC_2_CHW(image)
    img_chw = np.float32(img_chw)/255.0
    means, stds = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    img_chw = img_normalize(img_chw, means, stds)
    img_chw = np.expand_dims(img_chw, axis=0) # Making batch size of 1

    # image = image.convert('RGB')
    '''transform = T.Compose([
                           T.Resize((256, 256)),
                           T.ToTensor(),
                           T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                           ])
    tr_img = transform(image)
    print('### Transformed Image size : {}'.format(tr_img.shape))
    pixels = asarray(image)
    # confirm pixel range is 0-255
    print('Data Type: %s' % pixels.dtype)
    print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))
    # convert from integers to floats
    pixels = pixels.astype('float32')
    # normalize to the range 0-1
    pixels /= 255.0
    # confirm the normalization
    print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))
    print(pixels.shape)
    
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()'''

    # compute ONNX Runtime output prediction
    # ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(tr_img.unsqueeze(0))}
    ort_inputs = {ort_session.get_inputs()[0].name: img_chw}
    ort_outs = ort_session.run(None, ort_inputs)

    np.array(ort_outs).shape

    ort_outs = np.array(ort_outs[0][0])

    THRESHOLD = 0.8
    (OUT_HEIGHT, OUT_WIDTH) = (64,64)
    OUT_SHAPE = (OUT_HEIGHT, OUT_WIDTH)
    image_p = image 
    pose_layers = ort_outs

    from operator import itemgetter
    get_keypoints = lambda pose_layers: map(itemgetter(1, 3), [cv2.minMaxLoc(pose_layer) for pose_layer in pose_layers])

    key_points = list(get_keypoints(pose_layers=pose_layers))

    print(key_points)


    is_joint_plotted = [False for i in range(len(JOINTS))]

    print(is_joint_plotted)
    for pose_pair in POSE_PAIRS:
        from_j, to_j = pose_pair

        from_thr, (from_x_j, from_y_j) = key_points[from_j]
        to_thr, (to_x_j, to_y_j) = key_points[to_j]
        
        IMG_HEIGHT, IMG_WIDTH, _ = image_p.shape

        from_x_j, to_x_j = from_x_j * IMG_WIDTH / OUT_SHAPE[0], to_x_j * IMG_WIDTH / OUT_SHAPE[0]
        from_y_j, to_y_j = from_y_j * IMG_HEIGHT / OUT_SHAPE[1], to_y_j * IMG_HEIGHT / OUT_SHAPE[1]

        from_x_j, to_x_j = int(from_x_j), int(to_x_j)
        from_y_j, to_y_j = int(from_y_j), int(to_y_j)

        if from_thr > THRESHOLD and not is_joint_plotted[from_j]:
            # this is a joint
            cv2.ellipse(image_p, (from_x_j, from_y_j), (4, 4), 0, 0, 360, (255, 255, 255), cv2.FILLED)
            is_joint_plotted[from_j] = True

        if to_thr > THRESHOLD and not is_joint_plotted[to_j]:
            # this is a joint
            cv2.ellipse(image_p, (to_x_j, to_y_j), (4, 4), 0, 0, 360, (255, 255, 255), cv2.FILLED)
            is_joint_plotted[to_j] = True

        if from_thr > THRESHOLD and to_thr > THRESHOLD:
            # this is a joint connection, plot a line
            cv2.line(image_p, (from_x_j, from_y_j), (to_x_j, to_y_j), (255, 74, 0), 3)
            
    """Pretty similar results !? eh ?"""
    print('### HPE Image shape is : {}'.format(image_p.shape))
    #Image.fromarray(cv2.cvtColor(image_p, cv2.COLOR_RGB2BGR))

    #im = transform_image(image_bytes)
    return image_p # hpe image

def main_handler(event, context):
    try:
        content_type_header = event['headers']['content-type']
        body = base64.b64decode(event["body"])
        print('### BODY Loaded')
        picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
        hpe_image = get_hpe_image(picture.content)
        print('### HPE Done!!!')
        print(type(hpe_image))

        if hpe_image is None:
            image_response = ''
        else:
            image_response = img_to_base64(hpe_image)

        print(image_response)

        filename = picture.headers[b'Content-Disposition'].decode().split(';')[1].split('=')[1]
        if len(filename) < 4:
            filename = picture.headers[b'Content-Disposition'].decode().split(';')[2].split('=')[1]

        return {
            "statusCode": 200,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True

            },
            "body": json.dumps({'file': filename.replace('"', ''), 'imagebytes':  image_response})
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