import sys
import os
import torch
from torch import nn
import h5py

from filelock import FileLock

class FeatureStorage:
    def __init__(self, file_path):
        os.makedirs("./features_storage/locks", exist_ok=True)

        self.file_path = f"./features_storage/{file_path}"
        self.features = {}
        lock_file = f"./features_storage/locks/{file_path}.lock"
        # self.lock = FileLock(lock_file)
        self.load_all_features()
        
    def load_all_features(self):
        self.features = {}
        if os.path.exists(self.file_path):
            with h5py.File(self.file_path, 'r', swmr=True) as f:
                for key in f.keys():
                    self.features[key] = torch.tensor(f[key][:]).cuda()
        

    def save_features(self, features, indices):
        # with self.lock:
        with h5py.File(self.file_path, 'a') as f:
            for idx, feature in zip(indices, features):
                try:
                    f.create_dataset(str(idx), data=feature.cpu().detach().numpy())
                    self.features[str(idx)] = feature.cuda()
                except:
                    pass

    def load_features(self, indices):
        features = []
        for idx in indices:
            features.append(self.features[str(idx)])
        return torch.stack(features)
    
    def indices(self):
        if not os.path.exists(self.file_path):
            return set()
        with h5py.File(self.file_path, 'r', swmr=True) as f:
            return set(f.keys())
        
class Preceptor(nn.Module):
    def __init__(self, name, dataset, freeze=True):
        super(Preceptor, self).__init__()
        
    def forward(self, x):
        raise NotImplementedError
    
    def update(self, minibatches, unlabeled=None):
        all_x = []
        for x, y in minibatches:
            all_x.extend(x)
        self.forward(all_x)