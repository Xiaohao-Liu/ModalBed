import os
import requests
import gdown

### utils >>>

def download_file(url, destination):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(destination, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        print(f"File downloaded successfully: {destination}")
    else:
        print(f"Failed to download file. HTTP Status Code: {response.status_code}")

### <<< utils

def dataset_download_msr_vtt(data_dir):
    os.system(f"huggingface-cli download AlexZigma/msr-vtt --repo-type dataset --local-dir {os.path.join(data_dir, 'msr-vtt')}")
    
def dataset_download_nyud_v2(data_dir):
    os.system(f"huggingface-cli download sayakpaul/nyu_depth_v2 --repo-type dataset --local-dir {os.path.join(data_dir, 'nyud-v-2')}")
    
def dataset_download_vggsound(data_dir):
    os.system(f"huggingface-cli download Loie/VGGSound --repo-type dataset --local-dir {os.path.join(data_dir, 'vggsound')}")


def perceptor_download_imagebind(data_dir):
    os.system(f"git clone https://github.com/facebookresearch/ImageBind {data_dir}")
    os.rename(os.path.join(data_dir, "ImageBind"), os.path.join(data_dir, "imagebind"))
    
    download_file('https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth', os.path.join(data_dir, 'imagebind_huge.pth'))

def perceptor_download_unibind(data_dir):
    os.system(f"git clone https://github.com/qc-ly/UniBind {data_dir}")
    
    os.rename(os.path.join(data_dir, "UniBind"), os.path.join(data_dir, "unibind"))
    
    gdown.download("https://drive.google.com/u/0/uc?id=1Dgmj7ajdoT8hYHobJQfgIuB4CbVRoKUn&export=download&confirm=t",
                   os.path.join(data_dir, "unibind.pt")) # pre-trained weights 
        
    gdown.download_folder("https://drive.google.com/drive/folders/1aQ654WO9jFuK6bqz2YqhAVrT0vvXPiJW",
                   output=os.path.join(data_dir, "unibind_centre_embs")) # center_embeddings
    

def perceptor_download_languagebind(data_dir):
    os.system(f"git clone https://github.com/PKU-YuanGroup/LanguageBind/languagebind {data_dir}")
    os.rename(os.path.join(data_dir, "LanguageBind"), os.path.join(data_dir, "languagebind"))
    
    os.pardir(os.path.join(data_dir, "languagebind", "ckpts"))
    
    os.makedirs(os.path.join(data_dir, "languagebind", "ckpts"), exist_ok=True)
    
    os.system(f"huggingface-cli download LanguageBind/LanguageBind_Video_FT --local-dir {os.path.join(data_dir, 'languagebind', 'languagebind', 'ckpts/Video_FT')}")
    os.system(f"huggingface-cli download LanguageBind/LanguageBind_Audio_FT --local-dir {os.path.join(data_dir, 'languagebind', 'languagebind', 'ckpts/Audio_FT')}")
    os.system(f"huggingface-cli download LanguageBind/LanguageBind_Depth --local-dir {os.path.join(data_dir, 'languagebind', 'languagebind', 'ckpts/Depth')}")
    os.system(f"huggingface-cli download LanguageBind/LanguageBind_Thermal --local-dir {os.path.join(data_dir, 'languagebind', 'languagebind', 'ckpts/Thermal')}")
    os.system(f"huggingface-cli download LanguageBind/LanguageBind_Image --local-dir {os.path.join(data_dir, 'languagebind', 'languagebind', 'ckpts/Image')}")


def perceptor_download(perceptor):
    os.makedirs("modalEncoder", exist_ok=True)
    data_dir = "modalEncoder"
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