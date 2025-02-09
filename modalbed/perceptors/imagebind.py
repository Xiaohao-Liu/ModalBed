from modalbed.datasets import ModalityType
from modal_encoder.model import data, load_model, get_embed_dim
from .base import FeatureStorage, Preceptor
import torch

from .utils import make_modalityLoader, load_and_transform_depth_data, make_modalityMap

modalityLoader = make_modalityLoader(
    {
        ModalityType.TEXT: data.load_and_transform_text,
        ModalityType.VISION: data.load_and_transform_vision_data,
        ModalityType.AUDIO: data.load_and_transform_audio_data,
        ModalityType.VIDEO: data.load_and_transform_video_data,
        ModalityType.DEPTH: load_and_transform_depth_data,
    }
)

modalityMap = make_modalityMap(
    {
        ModalityType.VIDEO: ModalityType.VISION,
        ModalityType.TACTILE: ModalityType.VISION,
    }
)


class ImagebindPreceptor(Preceptor):
    def __init__(self, dataset="msrvtt", freeze=True, feature_retrieval=False):
        super(ImagebindPreceptor, self).__init__(dataset, freeze)
        self.n_outputs = get_embed_dim("imagebind")

        if feature_retrieval:
            self.model = (
                lambda x: {key: torch.randn(len(x[key]), self.n_outputs) for key in x}
            )  # avoid some errors. please prepara the features before the training, so that such function will not be called
        else:
            if freeze:
                model = load_model("imagebind").to("cuda")  # mannuall set to cuda
                self.__dict__["model"] = model  # discard the registration of parameters
                for param in self.model.parameters():
                    param.requires_grad = False
            else:
                self.model = load_model("imagebind")
        self.dataset = dataset
        self.feature_storage = FeatureStorage(
            f"{dataset}_imagebind.h5"
        )  # adopt h5 file to store/retrieve features
        self.existing_indices = self.feature_storage.indices()

    def forward(self, x):
        # x is a minibatch of data
        # device = self.model.device
        device = "cuda"
        datas = {m.value: [] for m in ModalityType}
        reterived_indices = []
        embed_pos = []  # (modal, pos)

        for i in x:
            modal = i["modal"]
            data = i["data"]  # caption for text, path for others
            id = i["id"]
            indice = "_".join([modal, id])
            is_stored = indice in self.existing_indices

            if not is_stored:
                datas[modal].append(data)
                pos = len(datas[modal]) - 1
            else:
                reterived_indices.append(indice)
                modal = "reterived"
                pos = len(reterived_indices) - 1

            embed_pos.append(
                (
                    modal,
                    pos,
                    indice,
                )
            )

        # inputs
        features_m = {}
        for m_type in datas:
            if len(datas[m_type]) == 0:
                continue
            inputs = {
                modalityMap(m_type): modalityLoader(m_type)(datas[m_type], device)
            }
            features_m[m_type] = self.model(inputs)[modalityMap(m_type)]

        if len(reterived_indices) > 0:
            features_m["reterived"] = self.feature_storage.load_features(
                reterived_indices
            )

        # reorganize the output

        features = []
        store_indices = []
        store_features = []
        for m, pos, indice in embed_pos:
            features.append(features_m[m][pos].to(device))
            if m != "reterived":
                store_indices.append(indice)
                store_features.append(features_m[m][pos])

        if len(store_indices) > 0:  # store new features (update)
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
