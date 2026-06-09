import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class emotionDataset(Dataset):
    def __init__(self,dataframe,transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        
        self.emotions = self.dataframe['emotion'] 
        self.pixels = self.dataframe['pixels']        
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        label = self.emotions.iloc[index]
        features = self.pixels.iloc[index]
        
        feature_matrix = features.reshape(48,48).astype(np.uint8)
        image = Image.fromarray(feature_matrix,mode='L')
        
        label = torch.tensor(label,dtype=torch.long)
        
        if self.transform:
            image = self.transform(image)
        else:
            default_transform = torch.ToTensor()
            image = default_transform(image)
        
        return  image,label