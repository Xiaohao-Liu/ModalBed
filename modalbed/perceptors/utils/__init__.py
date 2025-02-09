from collections.abc import Callable
from pathlib import Path
from typing import Mapping, cast
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch._prims_common import DeviceLikeType
from modal_encoder import data
from modalbed.datasets import ModalityType


def load_and_transform_depth_data(
    depth_paths: str | None, device: "DeviceLikeType", *, repeat: bool = False
) -> torch.Tensor | None:
    if depth_paths is None:
        return None
    device = torch.device(device)

    depth_outputs = list[torch.Tensor]()
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
        with Path(depth_path).open("rb") as fopen:
            image = Image.open(fopen)

        np_image = np.array(image.convert("L"), dtype=np.float32) / 255.0

        disparity = cast(torch.Tensor, data_transform(Image.fromarray(np_image))).to(
            device
        )
        if repeat:
            disparity = disparity.repeat(3, 1, 1)  # 3 channels

        depth_outputs.append(disparity)

    return torch.stack(depth_outputs, dim=0)


def make_modalityLoader(
    mapping: Mapping[ModalityType, Callable],
) -> Callable[[ModalityType], Callable]:
    def modalityLoader(x: ModalityType) -> Callable:
        return mapping.get(x, data.load_and_transform_vision_data)

    return modalityLoader


def make_modalityMap(
    mapping: Mapping[ModalityType, ModalityType | str],
) -> Callable[[ModalityType], ModalityType | str]:
    def modalityMap(x: ModalityType) -> ModalityType | str:
        return mapping.get(x, x)

    return modalityMap
