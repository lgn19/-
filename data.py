import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

transform2=transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化图像
])

class MyDataset(Dataset):
    def __init__(self,path1,path2,path3):
        self.path_seg=path1
        self.name_seg=os.listdir(os.path.join(path1,''))
        self.path_img=path2
        self.name_img=os.listdir(os.path.join(path2,''))
        self.path_w=path3
        self.name_w=os.listdir(os.path.join(path3,''))

    def __len__(self):
        return len(self.name_seg)

    def __getitem__(self, index):
        segment_name=self.name_seg[index]
        segment_path=os.path.join(self.path_seg,'',segment_name)
        img_name=self.name_img[index]
        img_path=os.path.join(self.path_img,'',img_name)
        w_name=self.name_w[index]
        w_path=os.path.join(self.path_w,'',w_name)


        segment_image=Image.open(segment_path).convert("L")
        image=Image.open(img_path).convert('RGB')
        w=Image.open(w_path).convert("L")
        return transform(image),transform2(segment_image),transform2(w)



if __name__ == '__main__':
    data=MyDataset('/Users/apple/Desktop/李广能/unet3.0/unet/data/SegmentationClass','/Users/apple/Desktop/李广能/unet3.0/unet/data/JPEGImages',"/Users/apple/Desktop/李广能/unet3.0/unet/data/权重")
    print(data[0].shape)
    print(data[0].shape)



