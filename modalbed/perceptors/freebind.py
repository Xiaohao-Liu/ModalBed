import torch
from modalbed.datasets import ModalityType
from modal_encoder.model import load_model, get_embed_dim
from .base import FeatureStorage, Preceptor


class FreebindPreceptor(Preceptor):
    def __init__(self, dataset="msrvtt", freeze=True, feature_retrieval=False):
        super(FreebindPreceptor, self).__init__(dataset, freeze)
        self.n_outputs = get_embed_dim("freebind")

        if feature_retrieval:
            self.model = (
                lambda x: {key: torch.randn(len(x[key]), self.n_outputs) for key in x}
            )  # avoid some errors. please prepara the features before the training, so that such function will not be called
        else:
            if freeze:
                model = load_model("freebind").to("cuda")  # mannuall set to cuda
                self.__dict__["model"] = model  # discard the registration of parameters
                for param in self.model.parameters():
                    param.requires_grad = False
            else:
                self.model = load_model("freebind")
        self.dataset = dataset
        self.feature_storage = FeatureStorage(
            f"{dataset}_freebind.h5"
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
        features_m = {
            m_type: self.model(value)
            for m_type, value in datas.items()
            if len(value) > 0
        }

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
