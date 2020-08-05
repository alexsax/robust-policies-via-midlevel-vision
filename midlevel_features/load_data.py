import os
import torch
import pandas as pd
import numpy as np
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms, utils
import glob
import random
from PIL import Image
import pathlib

class RLBenchEnvsDataset(Dataset):
    """RLBench dataset, spanning multiple envs"""

    def __init__(self, root_dir, task_name="normal", transform = None, transform_target= None):
        """
        Args:
            root_dir (string): Directory with the data. Expecting the format
            /{env} 
                /record_{###}
                    /frame{######}_{type}.png

            task_name (str):  one of ['rgb', 'normal', 'depth', 'sobel']
        """
        self.root_dir = root_dir
        self.files = [str(x) for x in list(pathlib.Path(self.root_dir).rglob("*_rgb.png"))]
        print("Dataset has", len(self.files), "unique images")

        self.task_name = task_name

        self.transform = transform
        
        self.transform_target = transform_target


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        im_path = self.files[idx]
        def load(path, task):
            img_arr = np.array(io.imread(path.replace('rgb',task)))
            return Image.fromarray(img_arr)
        sample = {}
        for task in ['rgb', self.task_name]:
            #special case: if the task is autoencoder, the label is just rgb. 
            if self.task_name == "autoencoder":
                sample[task] = load(im_path, "rgb")
            else:
                sample[task] = load(im_path, task)        
        if self.task_name != "segment_img":
            try:
                sample['segment_img'] = load(im_path, 'segment_img')
            except:
                # No segment img exists
                pass
        seed = np.random.randint(2147483647)
        if self.transform is not None:
            random.seed(seed)
            sample['rgb'] = self.transform(sample['rgb'])
        if self.transform_target is not None:
            random.seed(seed)
            sample[self.task_name] = self.transform_target(sample[self.task_name])
        try:
            if self.task_name != "segment_img":
                random.seed(seed)
                sample["segment_img"] = self.transform_target(sample['segment_img'])
        except:
            pass
        #rescale the image[tensor] here
        sample['rgb'] = sample['rgb'] * 2 - 1

        # Always return segment img mask, for use in the loss. 
        #[0] *255 undoes the .totensor() transfomration
        try:
            sample['segment_img'] =  np.array(sample['segment_img'][0] * 255)
        except:
            pass
        if self.task_name == 'segment_semantic':
            #*255 undoes the .totensor() transfomration
            sample['segment_semantic'] = np.array(sample['segment_semantic'][0] * 255)

        # for pix2pix tasks, scales targets to between -1 and 1, since the NN has a tanh output layer usually. 
        pix2pix_tasks = ['depth', 'normal', 'sobel', "autoencoder", "denoise"]
        if self.task_name in pix2pix_tasks:
            sample[self.task_name] = sample[self.task_name] * 2 - 1
        return sample


# adapted from https://discuss.pytorch.org/t/using-imagefolder-random-split-with-multiple-transforms/79899/3
trans = transforms.Compose([
    transforms.RandomCrop(256),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    ])

trans_target = transforms.Compose([
    transforms.RandomCrop(256),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    ])

transNoAugment = transforms.Compose([ transforms.CenterCrop(256),
                                  transforms.ToTensor(),
                                  ])

transNoAugment_target = transforms.Compose([
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    ])

def genDS(batch_size=128, num_workers=4, task_name='normal', path='fill in path'):
    task_name = task_name
    augmented = RLBenchEnvsDataset(path,task_name=task_name,transform=trans, transform_target=trans_target)
    nonaugmented = RLBenchEnvsDataset(path,task_name=task_name,transform=transNoAugment, transform_target=transNoAugment_target)

    # Create the index splits for training, validation and test
    train_size = 0.8
    num_train = len(augmented)
    indices = list(range(num_train))
    split = int(np.floor(train_size * num_train))
    split2 = int(np.floor((train_size+(1-train_size)/2) * num_train))
    np.random.shuffle(indices)
    train_idx, valid_idx, test_idx = indices[:split], indices[split:split2], indices[split2:]

    traindata = Subset(augmented, indices=train_idx)
    valdata = Subset(nonaugmented, indices=valid_idx)
    testdata = Subset(nonaugmented, indices=test_idx)


    trainLoader = torch.utils.data.DataLoader(traindata, batch_size=batch_size, 
                                            num_workers=num_workers, drop_last=True)
    valLoader = torch.utils.data.DataLoader(valdata, batch_size=batch_size, 
                                            num_workers=num_workers, drop_last=True)
    testLoader = torch.utils.data.DataLoader(testdata, batch_size=batch_size,
                                            num_workers=num_workers, drop_last=True)
    return trainLoader, valLoader, testLoader