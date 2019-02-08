import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
from PIL import Image
import os 

class TerrainDataset(Dataset): 
    '''
    Image data set with optional transform. 
    '''

    def __init__(self, transform=transforms.ToTensor()):
        '''
        Retrieves image file names and labels. 
        '''
        # Directory paths 
        dir_path = 'data/labelled/safe_'
        categories = [1, 2, 3, 4, 5]
        # Get number of images per subdirectory 
        num_images = [len([n for n in os.listdir(f'{dir_path}{i}')]) - 1 for i 
            in categories]

        self.transform = transform

        self.image_names = list()
        self.image_labels = list()
        # Save filenames and labels of each image
        for cat_num, total in zip(categories, num_images): 
            for img_num in range(total): 
                self.image_names.append(f'{dir_path}{cat_num}/{img_num}.tiff')
                self.image_labels.append(torch.zeros(len(categories) - 1))
                # Ordinal encoding 
                if cat_num != 1: 
                    self.image_labels[-1][:cat_num - 1] = 1
        for n in self.image_names: 
            if self.transform(Image.open(n)).shape[0] == 4: 
                print(n)

    def __getitem__(self, ind): 
        '''
        Returns image, label pair. 
        '''
        img = Image.open(self.image_names[ind])

        if self.transform is not None: 
            img = self.transform(img)

        if not isinstance(img, torch.Tensor):
            img = transforms.ToTensor(img) 

        return img, self.image_labels[ind]

    def __len__(self): 
        '''
        Returns data set length.
        '''
        return len(self.image_names)


def create_loaders(split=[0.7, 0.2, 0.1], transform=transforms.ToTensor(),
    batch_size=16, seed=None): 
    ''' 
    Create DataLoader for training, validation, test. 

    Parameters
    ----------
    split : list, optional
        List of floats summing to 1 for training, validation, 
        test split.
    transform : torchvision.transforms, optional
        A transform to apply to images. 

    Returns
    -------
    DataLoader, DataLoader, DataLoader
    '''
    # Validate split argument
    if (not all([isinstance(i, (float, int)) for i in split])) or \
        round(sum(split), 5) != 1:
        print(f'Split of {split} invalid - set to default split.')
        split = [0.7, 0.2, 0.1]

    # Load data, inds, and shuffle
    data = TerrainDataset(transform)
    inds = [i for i in range(len(data))]
    np.random.seed(seed)
    np.random.shuffle(inds)

    # Get samplers
    train_split = int(split[0] * len(inds))
    val_split = int(len(inds) - split[2] * len(inds))
    sample_inds = [inds[:train_split],
        inds[train_split:val_split], inds[val_split:]]
    samplers = [SubsetRandomSampler(ind) for ind in sample_inds]

    # Create DataLoaders
    loaders = [DataLoader(data, batch_size, sampler=sampler, num_workers=4) for
        sampler in samplers]

    return tuple(loaders)
    
create_loaders()
