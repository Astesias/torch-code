import os
from PIL import Image
import numpy as np
from pysl import easy_show_img,path2filename,backref_dict
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms

global TYPES,SIZE
TYPES={
        'apple':0,
        'banana':1,
        'bean':2,
        'cabbage':3,
        'chili':4,
        'corn':5,
        'cucumber':6,
        'durian':7,
        'eggplant':8,
        'grape':9,
        'orange':10,
        'peanut':11,
        'potato':12,
        'radish':13,
        'rice':14
       }
SIZE=224

class Loader(Dataset):
    def __init__(self,path,):
        self.path=path 
        files=os.listdir(self.path)
        self.files=[os.path.join(path,i) for i in files ]
        self.transform=transforms.Compose([
                             transforms.Resize((SIZE,SIZE)),
                             # transforms.CenterCrop(112),
                             transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
                             transforms.RandomAffine(10),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.82452958 ,0.77630531 ,0.67329322],
                                                 std=[0.19371124 ,0.23563428 ,0.32667516],)
                                         ])
    def __getitem__(self, index):
        file=self.files[index]
        image=Image.open(file)
        image=self.transform(image)
        label=TYPES[path2filename(file).split(' ')[-2]]
        return image,label
    def __len__(self):
        return len(self.files)

if os.name=='nt':
    num_workers=0
else:
    num_workers=2
    
dataloader_train=DataLoader(Loader('../dataset/train'),batch_size=56,shuffle=True,num_workers=0)
dataloader_test=DataLoader(Loader('../dataset/test'),batch_size=10,shuffle=False,num_workers=0)


def index2types(index):
    for i in list(TYPES.items()):
        if i[1]==index:
            return i[0]

if __name__ == '__main__':
    for imgs,targets in dataloader_train:
        for i in range(100):
            img=np.array(imgs[i])
            print(img.shape)
            typ=int(targets[i])
            print(backref_dict(TYPES,k2v=True).backref(typ))
            easy_show_img(img,transpose=(1,2,0),bgr=True)
    

 

