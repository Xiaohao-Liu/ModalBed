import os
from pathlib import Path
import requests
import gdown
import shutil
### utils >>>


def download_file(url, destination):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(destination, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        print(f"File downloaded successfully: {destination}")
    else:
        print(f"Failed to download file. HTTP Status Code: {response.status_code}")


### <<< utils


def dataset_download_msr_vtt(data_dir):
    os.system(
        f"huggingface-cli download AlexZigma/msr-vtt --repo-type dataset --local-dir {os.path.join(data_dir, 'msr-vtt')}"
    )


def dataset_download_nyud_v2(data_dir):
    os.system(
        f"huggingface-cli download sayakpaul/nyu_depth_v2 --repo-type dataset --local-dir {os.path.join(data_dir, 'nyud-v-2')}"
    )


def dataset_download_vggsound(data_dir):
    os.system(
        f"huggingface-cli download Loie/VGGSound --repo-type dataset --local-dir {os.path.join(data_dir, 'vggsound')}"
    )


def perceptor_download_imagebind(data_dir):
    os.system(f"git clone https://github.com/facebookresearch/ImageBind {data_dir}")
    shutil.move(
        os.path.join(data_dir, "ImageBind", "imagebind"),
        os.path.join(data_dir, "imagebind"),
    )
    shutil.rmtree(os.path.join(data_dir, "ImageBind"))
    download_file(
        "https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth",
        os.path.join(data_dir, "imagebind_huge.pth"),
    )


def perceptor_download_unibind(data_dir):
    os.system(f"git clone https://github.com/qc-ly/UniBind {data_dir}")

    os.rename(os.path.join(data_dir, "UniBind"), os.path.join(data_dir, "unibind"))

    gdown.download(
        "https://drive.google.com/u/0/uc?id=1Dgmj7ajdoT8hYHobJQfgIuB4CbVRoKUn&export=download&confirm=t",
        os.path.join(data_dir, "unibind.pt"),
    )  # pre-trained weights

    gdown.download_folder(
        "https://drive.google.com/drive/folders/1aQ654WO9jFuK6bqz2YqhAVrT0vvXPiJW",
        output=os.path.join(data_dir, "unibind_centre_embs"),
    )  # center_embeddings


def perceptor_download_pointbind(data_dir: str):
    os.system(f"git clone https://github.com/ZiyuGuo99/Point-Bind_Point-LLM {data_dir}")
    root_dir = Path(data_dir)
    (root_dir / "Point-Bind_Point-LLM").rename(root_dir / "pointbind")
    pointbind_dir = root_dir / "pointbind"
    (pointbind_dir / "ckpts").mkdir(exist_ok=True)
    (pointbind_dir / "Point-LLM").rename(pointbind_dir / "pointllm")
    gdown.download(
        "https://drive.google.com/file/d/1V9y3h9EPlPN_HzU7zeeZ6xBOcvU-Xj6h/view",
        str(root_dir / "pointbind_i2pmae.pt"),
    )  # pre-trained weights


def perceptor_download_freebind(data_dir: str):
    os.system(f"git clone https://github.com/zehanwang01/FreeBind {data_dir}")
    root_dir = Path(data_dir)
    (root_dir / "FreeBind").rename(root_dir / "freebind")
    os.system(
        f"huggingface-cli download Viglong/FreeBind --local-dir {root_dir / 'freebind'} --include 'checkpoints*'"
    )


def perceptor_download_languagebind(data_dir):
    os.system(
        f"git clone https://github.com/PKU-YuanGroup/LanguageBind/languagebind {data_dir}"
    )
    os.rename(
        os.path.join(data_dir, "LanguageBind"), os.path.join(data_dir, "languagebind")
    )

    os.pardir(os.path.join(data_dir, "languagebind", "ckpts"))

    os.makedirs(os.path.join(data_dir, "languagebind", "ckpts"), exist_ok=True)

    os.system(
        f"huggingface-cli download LanguageBind/LanguageBind_Video_FT --local-dir {os.path.join(data_dir, 'languagebind', 'languagebind', 'ckpts/Video_FT')}"
    )
    os.system(
        f"huggingface-cli download LanguageBind/LanguageBind_Audio_FT --local-dir {os.path.join(data_dir, 'languagebind', 'languagebind', 'ckpts/Audio_FT')}"
    )
    os.system(
        f"huggingface-cli download LanguageBind/LanguageBind_Depth --local-dir {os.path.join(data_dir, 'languagebind', 'languagebind', 'ckpts/Depth')}"
    )
    os.system(
        f"huggingface-cli download LanguageBind/LanguageBind_Thermal --local-dir {os.path.join(data_dir, 'languagebind', 'languagebind', 'ckpts/Thermal')}"
    )
    os.system(
        f"huggingface-cli download LanguageBind/LanguageBind_Image --local-dir {os.path.join(data_dir, 'languagebind', 'languagebind', 'ckpts/Image')}"
    )


def perceptor_download(perceptor: str):
    data_dir = "modal_encoder"
    Path("modal_encoder").mkdir(exist_ok=True)
    perceptor = perceptor.lower().replace("-", "_")
    if perceptor is not None:
        function_name = f"perceptor_download_{perceptor}"
        if function_name in globals():
            globals()[function_name](data_dir)
        else:
            print(f"Function '{function_name}' not found!")


def main(
    dataset: str = None,
    perceptor: str = None,
):
    if dataset is not None:
        data_dir = "./dataset"
        dataset = dataset.lower().replace("-", "_")
        function_name = f"dataset_download_{dataset}"
        if function_name in globals():
            globals()[function_name](data_dir)
        else:
            print(f"Function '{function_name}' not found!")

    if perceptor is not None:
        perceptor_download(perceptor)


if __name__ == "__main__":
    import fire

    fire.Fire(main)

# python -m modalbed.scripts.download --dataset="msr_vtt"
# python -m modalbed.scripts.download --perceptor="unibind"
