import os

from torch import nn,optim
import torch
from torch.utils.data import DataLoader
from data import *
from net import *
from torchvision.utils import save_image

# device=torch.device('mps' if torch.cuda.is_available() else 'cpu')
device=torch.device("mps")
weight_path='params/默认.pth'
data_path1='/Users/apple/Desktop/李广能/unet3.0/unet/data/SegmentationClass'
data_path2='/Users/apple/Desktop/李广能/unet3.0/unet/data/JPEGImages'
data_path3="/Users/apple/Desktop/李广能/unet3.0/unet/data/权重"
save_path='train_image'

def LOSS(pre,label,weight):
    # for j in weight:
    #     j=j**0.5
    return (weight*((pre-label)**2)).mean()


if __name__ == '__main__':
    data_loader=DataLoader(MyDataset(data_path1,data_path2,data_path3),batch_size=10,shuffle=False)
    net=UNet().to(device)
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print('successful load weight！')
    else:
        print('not successful load weight')

    opt=optim.NAdam(net.parameters(),lr=1e-2)

    epoch=1
    while True:
        for i,(image,segment_image,w) in enumerate(data_loader):
            # print(image.shape)
            image, segment_image,w=image.to(device),segment_image.to(device),w.to(device)
            loss_fun = nn.BCELoss(weight=w)
            # loss_fun = nn.MSELoss()
            # save_image(net(image[0]), 'short_time_img/{i}.bmp')
            # image__path="short_time_img/{i}.bmp"
            # out_image=doubval(image__path)
            out_image=net(image)
            # train_loss=loss_fun(out_image,segment_image)
            train_loss=LOSS(out_image,segment_image,w)

            opt.zero_grad()
            train_loss.backward()
            opt.step()

            if i%3==0:
                print(f'{epoch}-{i}-train_loss===>>{train_loss.item()}')

            if i%3==0:
                torch.save(net.state_dict(),weight_path)



            _image=image[0]
            _segment_image=segment_image[0]
            _out_image=out_image[0]


            # img=torch.stack([_image,_segment_image,_out_image],dim=0)
            # print(_out_image)
            save_image(_out_image,f'{save_path}/{i}.png')
        epoch+=1

