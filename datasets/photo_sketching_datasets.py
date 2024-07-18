import os.path
from glob import glob
import torchvision.transforms as transforms
# import torch
from PIL import Image
import torch.utils.data as data
import settings




class TestDirDataset(data.Dataset):
    def __init__(self, fine_size=256 , file_name=None, dataroot=None):
        self.fine_size = fine_size
        self.dataroot = dataroot
        if file_name:
            print("file_name", self.dataroot)
            self.list = [os.path.join(self.dataroot, file_name)]
        else:
            print("list", self.dataroot)
            self.list = []
            for ext in settings.IMG_EXTENSIONS:
                self.list.extend(glob(os.path.join(self.dataroot, '*' + ext)))

    def __getitem__(self, index):
        file_path = self.list[index]
        img = Image.open(file_path).convert('RGB')
        w, h = img.size
        fine_size = self.fine_size
        img = img.resize((fine_size, fine_size), Image.BICUBIC)
        img = transforms.ToTensor()(img)
        input = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)

        return {'input': input,'paths': file_path, 'w': w, 'h': h}

    def __len__(self):
        return len(self.list)

    def name(self):
        return 'TestDirDataset'

