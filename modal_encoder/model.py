import torch
from torch import nn
import torch.nn.functional as F
import pkg_resources
import os

from .unibind.models import PointBind_models as models
from .unibind.imagebind.imagebind_model import ModalityType


from .imagebind.models import imagebind_model
from .imagebind import data

from .languagebind.languagebind import LanguageBind as languagebind_model
from .languagebind.languagebind import to_device, transform_dict, LanguageBindImageTokenizer


class BindModel(nn.Module):
    def __init__(self):
        super(BindModel, self).__init__()

    def __forward__(self, inputs):
        raise NotImplementedError


class UniBind(BindModel):
    embed_dim = 1024
    def __init__(self, pretrain_weights):
        super(UniBind, self).__init__()
        self.backbone = models.PointBind_I2PMAE()
        self.backbone.load_state_dict(torch.load(pretrain_weights), strict=True)
        self.embed_dim = UniBind.embed_dim
        
    def forward(self, inputs):        
        embeddings = {}
        if "point" in inputs:
            pc_embeddings = self.backbone.encode_pc(inputs['point'])
            pc_embeddings = self.backbone.bind.modality_head_point(pc_embeddings)
            vision_embeddings = self.backbone.bind.modality_postprocessor_point(pc_embeddings)
            embeddings["point"] = vision_embeddings
            del inputs["point"]
        
        embeddings_2 = self.backbone.bind(inputs)
        
        for k, v in embeddings_2.items():
            embeddings[k] = v

        return embeddings
        
        # vision_embeddings = vision_embeddings / vision_embeddings.norm(dim=-1, keepdim=True)
        # return vision_embeddings
        
    @property
    def device(self):
        return next(self.parameters()).device
    
    
class ImageBind(BindModel):
    embed_dim = 1024
    def __init__(self, pretrain_weights):
        super(ImageBind, self).__init__()
        self.model = imagebind_model.ImageBindModel(
            vision_embed_dim=1280,
            vision_num_blocks=32,
            vision_num_heads=16,
            text_embed_dim=1024,
            text_num_blocks=24,
            text_num_heads=16,
            out_embed_dim=1024,
            audio_drop_path=0.1,
            imu_drop_path=0.7,
        )
        if pretrain_weights is not None:
            self.model.load_state_dict(torch.load(pretrain_weights))
        self.embed_dim = ImageBind.embed_dim
        
    
    def forward(self, inputs):
        return self.model(inputs)
    
    @property
    def device(self):
        return next(self.parameters()).device


import copy
class ImageBindModal(BindModel):
    embed_dim = 1024
    def __init__(self, modal_pre, modal_truck, path, pretrain_weights):
        super(ImageBindModal, self).__init__()
        self.model = imagebind_model.ImageBindModel(
            vision_embed_dim=1280,
            vision_num_blocks=32,
            vision_num_heads=16,
            text_embed_dim=1024,
            text_num_blocks=24,
            text_num_heads=16,
            out_embed_dim=1024,
            audio_drop_path=0.1,
            imu_drop_path=0.7,
        ) 
        self.model.load_state_dict(torch.load(pretrain_weights)) 
        # do not load the pretrained weights
        
        self.modal_pre = modal_pre
        self.modal_truck = modal_truck
        
        sup_modalities = ['vision', 'text', 'audio', 'thermal', 'depth', 'imu']
        redundant_modalities_pre = list(set(sup_modalities) - set([modal_pre]))
        redundant_modalities_truck = list(set(sup_modalities) - set([modal_truck]))

        # self.modal_name = modal_name
        
        # self.truck = copy.deepcopy(model.modality_trunks[modal_name])
        # self.preprocessor = copy.deepcopy(model.modality_preprocessors[modal_name])
        # self.head = copy.deepcopy(model.modality_heads[modal_name])
        # self.postprocessor = copy.deepcopy(model.modality_postprocessors[modal_name])
        
        for mod in sup_modalities:
            if mod in redundant_modalities_pre:
                del self.model.modality_preprocessors[mod]
                # del self.model.modality_trunks[mod]
                # del self.model.modality_heads[mod]
                # del self.model.modality_postprocessors[mod]
                
        for mod in sup_modalities:
            if mod in redundant_modalities_truck:
                # del self.model.modality_preprocessors[mod]
                del self.model.modality_trunks[mod]
                del self.model.modality_heads[mod]
                del self.model.modality_postprocessors[mod] 

        # for param in self.parameters():
        #     param.requires_grad = False
        
        feat_size = lambda x: {
            "vision": 1280,
            "text": 1024,
            "audio": 768,
            "depth": 384,
        }.get(x, 1280)
        
        if modal_pre in ["vision", "depth"]:
            self.projector = nn.Linear(feat_size(modal_pre), feat_size(modal_truck))
        
        self.embed_dim = ImageBindModal.embed_dim
        torch.cuda.empty_cache()
        
    
    def forward(self, inputs):
        outputs = {self.modal_pre: None}
        assert self.modal_pre in inputs, f"Modal {self.modal_pre} not in inputs"
        modality_value = inputs[self.modal_pre]
        outputs = {}
        reduce_list = (
            modality_value.ndim >= 5
        )  # Audio and Video inputs consist of multiple clips
        if reduce_list:
            B, S = modality_value.shape[:2]
            modality_value = modality_value.reshape(
                B * S, *modality_value.shape[2:]
            )
            


        if modality_value is not None:
            modality_value = self.model.modality_preprocessors[self.modal_pre](
                **{self.modal_pre: modality_value}
            )
            trunk_inputs = modality_value["trunk"]
            head_inputs = modality_value["head"]
            # import pdb; pdb.set_trace()
            if self.modal_pre in ["vision", "depth"]:
                trunk_inputs["tokens"] = self.projector(trunk_inputs["tokens"])
            modality_value = self.model.modality_trunks[self.modal_truck](**trunk_inputs)
            
            
            modality_value = self.model.modality_heads[self.modal_truck](
                modality_value, **head_inputs
            )
            modality_value = self.model.modality_postprocessors[self.modal_truck](
                modality_value
            )

            if reduce_list:
                modality_value = modality_value.reshape(B, S, -1)
                modality_value = modality_value.mean(dim=1)

            outputs[self.modal_pre] = modality_value

        return outputs
        # return self.model(inputs)
        
    
    @property
    def device(self):
        return next(self.parameters()).device


class LanguageBind(BindModel):
    embed_dim = 768
    transform_dict = transform_dict
    cache_dir = pkg_resources.resource_filename( __name__, "languagebind/languagebind/cache_dir")
    image_ckpt_path = pkg_resources.resource_filename(__name__, "languagebind/languagebind/ckpts/Image")
    def __init__(self, pretrain_weights):
        super(LanguageBind, self).__init__()
        clip_type = {
            'video': f'{pretrain_weights}/Video_FT',  # also LanguageBind_Video
            'audio': f'{pretrain_weights}/Audio_FT',  # also LanguageBind_Audio
            'thermal': f'{pretrain_weights}/Thermal',
            'image': f'{pretrain_weights}/Image',
            'depth': f'{pretrain_weights}/Depth',
        }
        
        self.model = languagebind_model(clip_type=clip_type, cache_dir=LanguageBind.cache_dir)

        self.modality_transform = {c: transform_dict[c](self.model.modality_config[c]) for c in clip_type.keys()}
        self.tokenizer = LanguageBindImageTokenizer.from_pretrained(LanguageBind.image_ckpt_path, cache_dir=os.path.join(LanguageBind.cache_dir, "tokenizer_cache_dir"))

        self.embed_dim = LanguageBind.embed_dim
        
    
    def forward(self, inputs):
        return self.model(inputs)
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    
def get_embed_dim(model: str="imagebind"):
    if model == "imagebind":
        return ImageBind.embed_dim
    elif model == "languagebind":
        return LanguageBind.embed_dim
    elif model == "unibind":
        return UniBind.embed_dim
    
    return None
def load_model(
    model: str = "imagebind",
    load_pretrain: bool = True,
    load_pretrain_path: str = None,
):
    if model == "imagebind":
        if load_pretrain:
            if load_pretrain_path is None:
                path = pkg_resources.resource_filename(
                    __name__, "imagebind_huge.pth"
                )
            else:
                path = load_pretrain_path
            model = ImageBind(path)
    
    elif model == "languagebind":
        if load_pretrain:
            if load_pretrain_path is None:
                path = pkg_resources.resource_filename(
                    __name__, "languagebind/languagebind/ckpts"
                )
            else:
                path = load_pretrain_path
        model = LanguageBind(path)
        
    elif model == "unibind":
        if load_pretrain:
            if load_pretrain_path is None:
                path = pkg_resources.resource_filename(
                    __name__, "unibind.pt"
                )
            else:
                path = load_pretrain_path
        model = UniBind(path)
        
    elif model.split("_")[0] == "strongMG":
        model_name, modal_pre, modal_truck = model.split("_")
        if load_pretrain:
            path = load_pretrain_path
        model = ImageBindModal(modal_pre, modal_truck, path, pretrain_weights=pkg_resources.resource_filename( __name__, "imagebind_huge.pth"))
        
    
    return model
        
if __name__ == "__main__":
    load_model("unibind")
    
    
