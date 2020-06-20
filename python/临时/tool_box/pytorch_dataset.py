import torch
from torch.utils.data  import Dataset
from PIL import Image
import torchvision.transforms as transforms


class SVMHDataset(Dataset):
    def __init__(self,img_path,img_label,transform=None):
        self.img_path=img_path
        self.img_label=img_label

        if transform is not None:
            self.transform=transform
        else:
            self.transform=None

    def __getitem__(self, index):
        img=Image.open(self.img_path[index]).convert('RGB')

        if self.transform is not None:
            img=self.transform(img)






