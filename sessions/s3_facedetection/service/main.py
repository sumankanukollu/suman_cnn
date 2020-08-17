import torch
from torchvision.models import resnet

# model = resnet.resnet34(pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
model = torch.load('D:\Praveen\sourcecontrol\github\praveenraghuvanshi\eva4-p2-group\Session-2\Assignment-2\model\modelnet_v2_44_77.pt')
model.eval()
print(model)


from torchvision import transforms
test_transforms = transforms.Compose([transforms.ToTensor(),
                                      #transforms.Normalize([0.485, 0.456, 0.406],
                                      #                     [0.229, 0.224, 0.225])
                                     ])



from torch.autograd import Variable

def predict_image(image):
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    #input = image
    input = input.to(device)
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    return index

from matplotlib import pyplot as plt

to_pil = torchvision.transforms.ToPILImage()
images, labels = get_random_images(5)
fig=plt.figure(figsize=(70,70))
for ii in range(len(images)):
    image = to_pil(images[ii])
    index = predict_image(image)
    #print(index)
    sub = fig.add_subplot(1, len(images), ii+1)
    res = int(labels[ii]) == index
    if res:
        sub.set_title(index)
    else:
        print('Misclassified Image :\n\tActual Label : {} and predicted : {}'.format(int(labels[ii]),index))
    plt.axis('off')
    plt.imshow(image)
plt.show()

#traced_model = torch.jit.trace(model, torch.randn(1,3,224,224))
#traced_model.save('mobilenet_v2.pt')

#print('****** Loading model')
#loadedModel = torch.jit.load('mobilenet_v2.pt')
#print('****** Loaded model')
#print(loadedModel)