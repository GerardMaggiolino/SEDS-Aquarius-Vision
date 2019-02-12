import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
from PIL import Image

class TerrainDataset(Dataset): 
    '''
    Image data set with optional transform. 
    '''

    def __init__(self, transform=transforms.ToTensor()):
        '''
        Retrieves image file names and labels. 
        '''
        self.img_path = 'data/labelled/processed/'
        self.transform = transform
        self.info_df = pd.read_csv('data/labelled/labels.csv', 
            header=None, names=['img', 'label'])
        self.categories = 5

    def __getitem__(self, ind): 
        '''
        Returns image, label pair. 
        '''
        name = self.info_df['img'].iloc[ind]
        img_name = f'{self.img_path}{name}'
        label = self.info_df['label'].iloc[ind]

        ordinal = torch.zeros(self.categories - 1)
        ordinal[0 : label - 1] = 1

        img = Image.open(img_name)
        img = self.transform(img)

        if not isinstance(img, torch.Tensor):
            img = transforms.ToTensor(img) 

        return img, ordinal

    def __len__(self): 
        '''
        Returns data set length.
        '''
        return len(self.info_df)


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
