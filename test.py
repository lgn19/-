import os

import torch
from PIL import Image

from net import *
# from unet.utils import keep_image_size_open
from data import *
from torchvision.utils import save_image


net=UNet()

weights='/Users/apple/Desktop/李广能/unet3.0/unet/params/默认.pth'
if os.path.exists(weights):
    net.load_state_dict(torch.load(weights))
    print('成功')
else:
    print('no loading')

# _input=input('please input JPEGImages path:')

# img=keep_image_size_open(_input)
img = Image.open("/Users/apple/Desktop/李广能/unet3.0/unet/train_image/0.png")
img_data=transform(img)
print(img_data.shape)
img_data=torch.unsqueeze(img_data,dim=0)
out=net(img_data)
save_image(out,'/Users/apple/Desktop/李广能/unet3.0/unet/test_image/test.png')

