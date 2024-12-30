import os
import torch
import json
from multiprocessing import Pool
import pandas as pd
from tqdm import tqdm
from torch.utils.data import TensorDataset


def extract_audio(param):
    video_path, audio_path = param
    if os.path.exists(audio_path):
        return True
    extract_audio_cmd = f"ffmpeg -i {video_path} -ab 160k -ac 2 -ar 44100 -vn {audio_path}  > /dev/null 2>&1"
    status = os.system(extract_audio_cmd)
    return status == 0

from types import SimpleNamespace

# follow imagebind
ModalityType = SimpleNamespace(
    VISION="vision", # "image" for languagebind, 
    TEXT="text", # # "language" for languagebind, 
    AUDIO="audio",
    THERMAL="thermal",
    DEPTH="depth",
    IMU="imu",
    VIDEO="video",
    TACTILE="tactile", # for tvl
    POINT="point", # for unibind
)

DATASETS = [
    "MSR_VTT",
    "VGGSound_S",
    "NYUDv2"
]

def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)

class MultipleModalityDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 8            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets) 

    
class MSR_VTT(MultipleModalityDataset):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = [ModalityType.VIDEO, ModalityType.AUDIO, ModalityType.TEXT]
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.datasets = []
        self.path = os.path.join(root, "msr-vtt/data/MSR-VTT/")
        reload_data = False
        self.input_shape = (3, 48,48) # no sense 
        self.modalities = {i:[] for i in MSR_VTT.ENVIRONMENTS}
        
        self.modalities_path = os.path.join(self.path, "modalities.json")
        
        if os.path.exists(self.modalities_path) and not reload_data:
            with open(self.modalities_path, "r") as f:
                self.modalities = json.load(f)
        else:
            self.load_modalities()
            with open(self.modalities_path, "w") as f:
                json.dump(self.modalities, f)
        
        self.num_classes = 20 # classes
        self.datasets = list(self.modalities.values())
    
    def load_modalities(self):
        id2label = {}  
        indices = set()
        os.makedirs(os.path.join(self.path, "audios"), exist_ok=True)
            
        def load_(path, data_path):
            with open(path, "r") as f:
                data = json.load(f)
                for item in data['videos']:
                    # {"category": 9, "url": "https://www.youtube.com/watch?v=9lZi22qLlEo", "video_id": "video0", "start time": 137.72, "end time": 149.44, "split": "train", "id": 0}
                    video_path = os.path.join(self.path, data_path, item["video_id"]+".mp4")
                    audio_path = os.path.join(self.path, "audios", item["video_id"]+".mp3")
                    
                    indice = ModalityType.VIDEO + "_" + item["video_id"]
                    if indice not in indices:
                        indices.add(indice)
                        self.modalities[ModalityType.VIDEO].append(
                            (
                                {"label":item["category"], 
                                "id":item["video_id"], 
                                "modal": ModalityType.VIDEO,
                                "data":video_path},
                                item["category"]
                            )
                            )
                    indice = ModalityType.AUDIO + "_" + item["video_id"]
                    if indice not in indices:
                        if extract_audio((video_path, audio_path)):
                            self.modalities[ModalityType.AUDIO].append(
                                (
                                    {"label":item["category"], 
                                    "id":item["video_id"], 
                                    "modal":ModalityType.AUDIO,
                                    "data":audio_path},
                                    item["category"]
                                )
                                )
                    id2label[item["video_id"]] = item["category"]
                for item in data['sentences']:
                    # {"caption": "two troops speak with one another", "video_id": "video140", "sen_id": 140198}
                    indice = ModalityType.TEXT + "_" + item["video_id"]
                    if indice not in indices:
                        indices.add(indice)
                        self.modalities[ModalityType.TEXT].append(
                            (
                                {
                                    "label": id2label[item["video_id"]],
                                    "id": item["video_id"],
                                    "modal": ModalityType.TEXT,
                                    "data": item["caption"]
                                },
                                id2label[item["video_id"]]
                            )
                        )
        load_(os.path.join(self.path, "train_val_annotation/train_val_videodatainfo.json"), "train_val_videos/TrainValVideo")
        load_(os.path.join(self.path, "test_annotation/test_videodatainfo.json"), "test_videos")

               
class NYUDv2(MultipleModalityDataset):
    CHECKPOINT_FREQ = 300
    N_STEPS = 10001           # Default, subclasses may override
    N_WORKERS = 8
    ENVIRONMENTS = [ModalityType.DEPTH, ModalityType.VISION, ModalityType.TEXT]
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.datasets = []
        self.path = os.path.join(root, "nyu-d-2/raw_data")
        
        reload_data = False
        
        self.input_shape = (3, 48,48) # no sense
        
        # load different modalities / environments
        self.modalities = {i:[] for i in NYUDv2.ENVIRONMENTS}
        self.classes = {}
        
        self.modalities_path = os.path.join(self.path, "modalities.json")
        
        if os.path.exists(self.modalities_path) and not reload_data:
            with open(self.modalities_path, "r") as f:
                self.modalities = json.load(f)
            with open(os.path.join(self.path, "classes.json"), "r") as f:
                self.classes = json.load(f)
        else:
            self.load_modalities()
            with open(self.modalities_path, "w") as f:
                json.dump(self.modalities, f)
            with open(os.path.join(self.path, "classes.json"), "w") as f:
                json.dump(self.classes, f)
        
        
        self.num_classes = len(self.classes) # classes
        self.datasets = list(self.modalities.values())
            
    def load_modalities(self):
        indices = set()
        
        def load_(mode="train"):
            path = os.path.join(self.path, f"{mode}")
            for folder in os.listdir(path):
                path2 = os.path.join(path, folder)
                label = folder.split("_")[0]
                if label not in self.classes:
                    self.classes[label] = len(self.classes)
                
                for i in os.listdir(path2):
                    path3 = os.path.join(path2, i)
                    id = folder + "_" + i
                    indice = ModalityType.DEPTH + "_" + str(id)
                    if indice not in indices:
                        indices.add(indice)
                    self.modalities[ModalityType.DEPTH].append(
                        (
                            {"label": label, 
                            "id": str(id), 
                            "modal": ModalityType.DEPTH,
                            "data":os.path.join(path3, "depth.png")},
                            self.classes[label] # label
                        )
                        )
                    indice = ModalityType.VISION + "_" + str(id)
                    if indice not in indices:
                        indices.add(indice)
                    self.modalities[ModalityType.VISION].append(
                        (
                            {"label": label, 
                            "id": str(id), 
                            "modal": ModalityType.VISION,
                            "data":os.path.join(path3, "rgb.png")},
                            self.classes[label] # label
                        )
                        )
                    
                    indice = ModalityType.TEXT + "_" + str(id)
                    if indice not in indices:
                        indices.add(indice)
                    self.modalities[ModalityType.TEXT].append(
                        (
                            {"label": label, 
                            "id": str(id), 
                            "modal": ModalityType.TEXT,
                            "data": str(label)},
                            self.classes[label] # label
                        )
                        )

        load_("train")
           

class VGGSound(MultipleModalityDataset):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = [ModalityType.VIDEO, ModalityType.AUDIO, ModalityType.TEXT]
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.datasets = []
        self.path = os.path.join(root, "vggsound/scratch/shared/beegfs/hchen/train_data/VGGSound_final/")
        self.csv_path = os.path.join(root, "vggsound/vggsound.csv")
        reload_data = False
        self.input_shape = (3, 48,48) # no sense 
        
        # load different modalities / environments
        self.modalities = {i:[] for i in VGGSound.ENVIRONMENTS}
        self.classes = {}
        
        self.modalities_path = os.path.join(self.path, "modalities.json")
        
        if os.path.exists(self.modalities_path) and not reload_data:
            with open(self.modalities_path, "r") as f:
                self.modalities = json.load(f)
            with open(os.path.join(self.path, "classes.json"), "r") as f:
                self.classes = json.load(f)
        else:
            self.load_modalities()
            with open(self.modalities_path, "w") as f:
                json.dump(self.modalities, f)
            with open(os.path.join(self.path, "classes.json"), "w") as f:
                json.dump(self.classes, f)
        
        self.num_classes = len(self.classes) # classes
        self.datasets = list(self.modalities.values())
        
    def extract_audio_from_video(self):
        
        data_pd = pd.read_csv(self.csv_path, names=["youtube_id","start_sec","label", "split"])
        
        tasks = []
        for i in data_pd.iterrows():
            youtube_id, start_sec, label, split = i[1]
            id = youtube_id + f"_{start_sec:06d}"
            video_path = os.path.join(self.path, "video", id+".mp4")
            audio_path = os.path.join(self.path, "audio", id+".mp3")
            
            tasks.append((video_path, audio_path))
        
        with Pool() as pool:
            pool.map(extract_audio, tasks)
    
    def load_modalities(self, max_samples=1e10):
        indices = set()
        os.makedirs(os.path.join(self.path, "audio"), exist_ok=True)
        self.extract_audio_from_video()

        data_pd = pd.read_csv(self.csv_path, names=["youtube_id","start_sec","label", "split"])
        count = 0
        for i in tqdm(data_pd.iterrows(), total=len(data_pd)):
            youtube_id, start_sec, label, split = i[1]
            if not label in self.classes:
                self.classes[label] = len(self.classes)
                
            id = youtube_id + f"_{start_sec:06d}"
            if not os.path.exists(os.path.join(self.path, "video", id+".mp4")):
                continue
            if count > max_samples:
                break
            count += 1
            video_path = os.path.join(self.path, "video", id+".mp4")
            audio_path = os.path.join(self.path, "audio", id+".mp3")
            indice = ModalityType.VIDEO + "_" + id
            if indice not in indices:
                indices.add(indice)
                self.modalities[ModalityType.VIDEO].append(
                    (
                        {"label":label, 
                        "id":id, 
                        "modal": ModalityType.VIDEO,
                        "data":video_path},
                        self.classes[label]
                    )
                    )
            indice = ModalityType.AUDIO + "_" + id
            if indice not in indices:
                if os.path.exists(audio_path):
                    self.modalities[ModalityType.AUDIO].append(
                        (
                            {"label":label, 
                            "id":id, 
                            "modal":ModalityType.AUDIO,
                            "data":audio_path},
                            self.classes[label]
                        )
                        )   
                    
            indice = ModalityType.TEXT + "_" + id
            if indice not in indices:
                self.modalities[ModalityType.TEXT].append(
                    (
                        {"label":label, 
                        "id":id, 
                        "modal":ModalityType.TEXT,
                        "data":label},
                        self.classes[label]
                    )
                    )   


class VGGSound_S(VGGSound):
    def load_modalities(self, max_samples=10000):
        return super().load_modalities(max_samples)


class Debug(MultipleModalityDataset):
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2
        self.datasets = []
        for _ in [0, 1, 2]:
            self.datasets.append(
                TensorDataset(
                    torch.randn(16, *self.INPUT_SHAPE),
                    torch.randint(0, self.num_classes, (16,))
                )
            )

