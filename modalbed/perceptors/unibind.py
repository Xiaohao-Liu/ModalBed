import sys
sys.path.append("modal_encoder")
import os
from modalbed.datasets import ModalityType
from modal_encoder.model import data, load_model, get_embed_dim
from .base import FeatureStorage, Preceptor
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from modal_encoder.unibind.utils.data_transform import load_and_transform_vision_data, load_and_transform_text, load_and_transform_audio_data, load_and_transform_thermal_data, load_and_transform_point_data, load_and_transform_video_data

def load_and_transform_depth_data(depth_paths, device):
    if depth_paths is None:
        return None
    device = torch.device(device)

    depth_outputs = []
    for depth_path in depth_paths:
        data_transform = transforms.Compose(
            [
                transforms.Resize(
                    224, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )
        with open(depth_path, "rb") as fopen:
            image = Image.open(fopen).convert("L")

        
        image = np.array(image, dtype=np.float32) / 255.0
        disparity = Image.fromarray(image)

        disparity = data_transform(disparity).to(device)

        depth_outputs.append(disparity)

    return torch.stack(depth_outputs, dim=0)


modalityLoader = lambda x:{
        ModalityType.TEXT: load_and_transform_text,
        ModalityType.VISION: load_and_transform_vision_data,
        ModalityType.AUDIO: load_and_transform_audio_data,
        ModalityType.VIDEO: load_and_transform_video_data,
        ModalityType.DEPTH: load_and_transform_depth_data,
        ModalityType.THERMAL: load_and_transform_thermal_data,
        ModalityType.POINT: load_and_transform_point_data,
    }.get(x, data.load_and_transform_vision_data)

modalityMap = lambda x:{
       ModalityType.VIDEO: ModalityType.VISION,
    }.get(x, x)

class UniBindPreceptor(Preceptor):
    def __init__(self, dataset="msrvtt", freeze=True, feature_retrieval=False):
        super(UniBindPreceptor, self).__init__(dataset, freeze)
        
        self.n_outputs = get_embed_dim("unibind")
        
        if feature_retrieval: 
            self.model = lambda x: {key: torch.randn(len(x[key]), self.n_outputs) for key in x} # avoid some errors. please prepara the features before the training, so that such function will not be called
        else:
            if freeze:
                model = load_model("unibind").to("cuda") # mannuall set to cuda
                self.__dict__["model"] = model # discard the registration of parameters
                for param in self.model.parameters():
                    param.requires_grad = False
            else:
                self.model = load_model("unibind")
        self.dataset = dataset
        
        
        self.feature_storage = FeatureStorage(f"{dataset}_unibind.h5") # adopt h5 file to store/retrieve features
        self.existing_indices = self.feature_storage.indices()
                
        
    def forward(self, x):
        # x is a minibatch of data
        device = "cuda"
        datas = {m:[] for m in ModalityType.__dict__.values()}
        reterived_indices = []
        embed_pos = [] # (modal, pos)
                
        for i in x:
            modal = i['modal']
            data = i['data'] # caption for text, path for others
            id = i['id']
            indice = "_".join([modal, id])
            is_stored = indice in self.existing_indices
            
            if not is_stored:
                datas[modal].append(data)
                pos = len(datas[modal])-1
            else:
                reterived_indices.append(indice)
                modal = "reterived"
                pos = len(reterived_indices)-1
                
            embed_pos.append((
                modal, 
                pos,
                indice,
                ))
        
        # inputs
        features_m = {}
        for m_type in datas:
            if len(datas[m_type]) == 0:
                continue
            inputs = {}
            inputs[modalityMap(m_type)] = modalityLoader(m_type)(datas[m_type], device)
                
            features_m[m_type] = self.model(inputs)[modalityMap(m_type)]
        
        if len(reterived_indices) > 0:
            features_m['reterived'] = self.feature_storage.load_features(reterived_indices)
        
        # reorganize the output
        
        features = []
        store_indices = []
        store_features = []
        for m, pos, indice in embed_pos:
            features.append(features_m[m][pos].to(device))
            if m != "reterived":
                store_indices.append(indice)
                store_features.append(features_m[m][pos])
        
        if len(store_indices) > 0: # store new features (update)
            self.feature_storage.save_features(store_features, store_indices)
            self.existing_indices = self.feature_storage.indices()

        features = torch.stack(features)
        # normalize
        features = features / features.norm(dim=-1, keepdim=True)
        
        return features
    
    def update(self, minibatches, unlabeled=None):
        all_x = []
        for x, y in minibatches:
            all_x.extend(x)
        self.forward(all_x)