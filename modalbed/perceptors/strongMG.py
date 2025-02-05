import sys
sys.path.append("modal_encoder")
import os
from modalbed.datasets import ModalityType
from modal_encoder.model import data, load_model
from .base import FeatureStorage, Preceptor
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from torchvision import transforms
from peft import PeftModel, LoraConfig, get_peft_model
from torch.cuda.amp import autocast


infonce_criterion = nn.CrossEntropyLoss()

def cl_loss_function(a, b, temp=0.2):
    a = nn.functional.normalize(a, dim=-1)
    b = nn.functional.normalize(b, dim=-1)
    logits = torch.mm(a, b.T)
    logits /= temp
    labels = torch.arange(a.shape[0]).to(a.device)
    return infonce_criterion(logits, labels)

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

        disparity = disparity.repeat(3, 1, 1) # 3 channels
        
        depth_outputs.append(disparity)

    return torch.stack(depth_outputs, dim=0)

from transformers import AutoTokenizer, T5EncoderModel
from transformers import AutoImageProcessor, ViTModel

text_preprocessor = AutoTokenizer.from_pretrained("google-t5/t5-small")

modalityLoader = lambda x:{
        ModalityType.TEXT: lambda x, device: text_preprocessor(x, padding=True, truncation=True, return_tensors="pt").input_ids.to(device),#data.load_and_transform_text,
        ModalityType.VISION: data.load_and_transform_vision_data,
        ModalityType.AUDIO: lambda x, device: data.load_and_transform_audio_data(x,device, num_mel_bins=224, target_length=224).mean(dim=2), # [bs, 3 , 1, H, W]
        ModalityType.VIDEO: lambda x, device: data.load_and_transform_video_data(x,device).mean(dim=1).mean(dim=2), # [bs, N_clips*3 , 3, N_frame, H, W]
        ModalityType.DEPTH: load_and_transform_depth_data,
    }.get(x, data.load_and_transform_vision_data)

modalityMap = lambda x:{
        ModalityType.VIDEO: ModalityType.VISION, 
        ModalityType.TACTILE: ModalityType.VISION,
    }.get(x, x)

class ContrastivePreceptor(nn.Module):
    embed_dim = 1024
    def __init__(self, dataset="msrvtt", freeze=True, modal=ModalityType.VISION, train=False):
        super(ContrastivePreceptor, self).__init__()
        
        self.train_mode = train
        self.modal_name = modal
        self.modal_name_imagebind_pre =modalityMap(modal)
        self.modal_name_imagebind_truck = "audio" if modal != "text" else "text"

        if modal == ModalityType.TEXT:
            self.preprocessor = AutoTokenizer.from_pretrained("google-t5/t5-small")
            self.encoder_model = T5EncoderModel.from_pretrained("google-t5/t5-small")
            self.model = lambda x: self.encoder_model(input_ids=x).last_hidden_state[:, 0, :]
            self.embed_dim = self.encoder_model.config.d_model
        else: 
            self.preprocessor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
            self.encoder_model = ViTModel.from_pretrained("google/vit-base-patch16-224")
            self.model = lambda x: self.encoder_model(pixel_values=x).last_hidden_state[:, 0, :]
            self.embed_dim = self.encoder_model.config.hidden_size
            
        self.dataset = dataset
        self.projector = nn.Linear(self.embed_dim, ContrastivePreceptor.embed_dim)
        
    def train_forward(self, x):
        device = "cuda"
        datas = []
        for i, _ in x:
            modal = i['modal']
            assert modal == self.modal_name
            data = i['data'] # caption for text, path for others
            id = i['id']
            indice = "_".join([modal, id])
            datas.append(data)

        data_1 = modalityLoader(self.modal_name)(datas, device) # use the data loader from imagebind
        
        random_tensor = torch.rand(data_1.shape).to(data_1.device)
        mask_bool = random_tensor < 0.3
        data_2 = data_1.masked_fill(mask_bool, 0)
        
        inputs = torch.cat([data_1, data_2], dim=0)
        
        with autocast():
            features = self.model(inputs)
        
        features = self.projector(features)

        return (features[:len(data_1)], features[len(data_1):])
                
    def forward(self, x):
        if self.train_mode:
            return self.train_forward(x)
        
        # x is a minibatch of data
        device = "cuda"
        datas = []
        for i, _ in x:
            modal = i['modal']
            assert modal == self.modal_name
            data = i['data'] # caption for text, path for others
            id = i['id']
            indice = "_".join([modal, id])
            datas.append(data)

        inputs = modalityLoader(self.modal_name)(datas, device).to(device) # use the data loader from imagebind
        
        with autocast():
            features = self.model(inputs)
        
        return self.projector(features)


from modalbed import datasets
import pkg_resources
import yaml

class StrongMGPreceptor(Preceptor):
    def __init__(self, dataset="MSR_VTT", freeze=True, feature_retrieval=False):
        super(StrongMGPreceptor, self).__init__(dataset, freeze)
        checkpoint_path_yaml = pkg_resources.resource_filename( __name__, "checkpoint_path.yaml")
        # yaml_data = yaml.load(checkpoint_path_yaml)
        with open(checkpoint_path_yaml, 'r') as file:
            yaml_data = yaml.safe_load(file)
        data = yaml_data[dataset]
        
        
        self.n_outputs = ContrastivePreceptor.embed_dim

        if feature_retrieval: 
            self.model = lambda x: {key: torch.randn(len(x[key]), self.n_outputs) for key in x} # avoid some errors. please prepara the features before the training, so that such function will not be called
        else:
            self.models = {}
            for modal in data:
                ckpt_path = data[modal]
                self.models[modal] = ContrastivePreceptor(dataset, freeze=False, modal=modal, train=False)
                # models[modal].load_state_dict(torch.load(ckpt_path)) # load weights
                lora_config = LoraConfig(
                    r=8,
                    target_modules=["q","k","v", "o", "wi", "wo"] if modal == "text" else ["query","key","value", "dense"] ,
                    lora_alpha=32,
                    lora_dropout=0.05,
                    modules_to_save=["projector"],
                )
                self.models[modal] = PeftModel.from_pretrained(self.models[modal], ckpt_path, config=lora_config, is_trainable=False).to("cuda")

                print(f"Loaded [{modal}] model from {ckpt_path}")
        self.dataset = dataset
        
        self.feature_storage = FeatureStorage(f"{dataset}_strongMG.h5") # adopt h5 file to store/retrieve features
        self.existing_indices = self.feature_storage.indices()
                
    def forward(self, x):
        device = "cuda"
        datas = {m:[] for m in ModalityType.__dict__.values()}
        reterived_indices = []
        embed_pos = [] # (modal, pos)
        
        for i in x:
            modal = i['modal']
            id = i['id']
            indice = "_".join([modal, id])
            is_stored = indice in self.existing_indices

            if not is_stored:
                datas[modal].append((i, None)) 
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
            # import pdb; pdb.set_trace()
            
            features_m[m_type] = self.models[m_type](datas[m_type])
        
        if len(reterived_indices) > 0:
            features_m['reterived'] = self.feature_storage.load_features(reterived_indices)
        
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

class CustomDataset(Dataset):
    """
    Custom dataset to handle the input data for the StrongMGPreceptor.
    """
    def __init__(self, data, modal):
        self.data = data
        self.modal = modal

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item
    
def collate_fn(batch):
    # import pdb; pdb.set_trace()
    return {"x": batch}

from transformers import Trainer, TrainingArguments

class Trainer2(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # 前向计算特征
        features = model(inputs["x"])
        # 计算对比损失
        features1, features2 = features
        # contrastive learning
        loss = cl_loss_function(features1, features2)
        return (loss, features) if return_outputs else loss
    
    def _save(self, output_dir= None, state_dict=None):
        super()._save(output_dir, state_dict)
        self.model.save_pretrained(output_dir)
        for i in [
            os.path.join(output_dir, "pytorch_model.bin"),
        ]:
            if os.path.exists(i):
                os.remove(i)
                
def train(dataset, modal, data_dir="./datasets"):
    if dataset in vars(datasets):
        dataset_model = vars(datasets)[dataset](data_dir, None, None)
    
    train_dataset = CustomDataset(dataset_model.modalities[modal], modal)
    
    preceptor = ContrastivePreceptor(dataset, freeze=False, modal=modal, train=True)

    # lora finetune
    lora_config = LoraConfig(
        r=8,
        target_modules=["q","k","v", "o", "wi", "wo"] if modal == "text" else ["query","key","value", "dense"] ,
        lora_alpha=32,
        lora_dropout=0.05,
        modules_to_save=["projector"],
    )
                
    preceptor_lora_model = get_peft_model(preceptor, lora_config)
    
    trainable_params = sum(p.numel() for p in preceptor.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in preceptor.parameters())

    print(f"Number of trainable parameters: {trainable_params/1e6}") # 4MB
    print(f"Number of total parameters: {total_params/1e6}") # 600MB
    

    if modal == "text":
        ds_modal = "text"
    else:
        ds_modal = "vision" # others
        
    batch_size = {"vision": 64, "text": 128}[ds_modal]
    
    training_args = TrainingArguments(
        output_dir=f"./strongMG_results/{dataset}/{modal}",          # Directory to save model checkpoints
        save_total_limit=5,             # Limit the total amount of checkpoints   
        save_strategy="steps",          # Save model every few steps
        save_steps=100,                 # Save steps
        per_device_train_batch_size=int(batch_size),  # Batch size per device
        num_train_epochs=5,             # Number of epochs
        logging_dir="./strongMG_logs",           # Directory for logs
        logging_steps=10,               # Log every 10 steps
        learning_rate=5e-5,             # Learning rate
        weight_decay=0.01,              # Weight decay
        report_to="none",                # Disable reporting to any logging service
    )
    
    trainer = Trainer2(
        model=preceptor_lora_model,  # Custom wrapper for StrongMGPreceptor
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,  # Add an evaluation dataset if available
        data_collator=collate_fn,
    )

    trainer.train()
    

if __name__ == "__main__":
    from fire import Fire
    Fire(train)
    
# CUDA_VISIBLE_DEVICES=0 python -m modalbed.perceptors.strongMG MSR_VTT video