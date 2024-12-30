# Following domainbed

"""
A command launcher launches a list of commands on a cluster; implement your own
launcher to add support for your cluster. We've provided an example launcher
which runs all commands serially on the local machine.
"""

import subprocess
import time
import torch
import os

def local_launcher(commands):
    """Launch commands serially on the local machine."""
    for cmd in commands:
        subprocess.call(cmd, shell=True)

def dummy_launcher(commands):
    """
    Doesn't run anything; instead, prints each command.
    Useful for testing.
    """
    for cmd in commands:
        print(f'Dummy launcher: {cmd}')

def multi_gpu_launcher(commands):
    """
    Launch commands on the local machine, using all GPUs in parallel.
    """
    print('WARNING: using experimental multi_gpu_launcher.')
    try:
        # Get list of GPUs from env, split by ',' and remove empty string ''
        # To handle the case when there is one extra comma: `CUDA_VISIBLE_DEVICES=0,1,2,3, python3 ...`
        available_gpus = [x for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',') if x != '']
    except Exception:
        # If the env variable is not set, we use all GPUs
        available_gpus = [str(x) for x in range(torch.cuda.device_count())]
    n_gpus = len(available_gpus)
    procs_by_gpu = [None]*n_gpus

    while len(commands) > 0:
        for idx, gpu_idx in enumerate(available_gpus):
            proc = procs_by_gpu[idx]
            if (proc is None) or (proc.poll() is not None):
                # Nothing is running on this GPU; launch a command.
                cmd = commands.pop(0)
                new_proc = subprocess.Popen(
                    f'CUDA_VISIBLE_DEVICES={gpu_idx} {cmd}', shell=True)
                procs_by_gpu[idx] = new_proc
                break
        time.sleep(1)

    # Wait for the last few tasks to finish before returning
    for p in procs_by_gpu:
        if p is not None:
            p.wait()
            
 
def manual_multi_gpu_launcher(commands):
    try:
        # Get list of GPUs from env, split by ',' and remove empty string ''
        # To handle the case when there is one extra comma: `CUDA_VISIBLE_DEVICES=0,1,2,3, python3 ...`
        available_gpus = [x for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',') if x != '']
    except Exception:
        # If the env variable is not set, we use all GPUs
        available_gpus = [str(x) for x in range(torch.cuda.device_count())]
        
    n_gpus = len(available_gpus)
    command_chunks = [commands[i::n_gpus] for i in range(n_gpus)]
    
    sh_file_dir = f"run_sh/{time.time()}"
    os.makedirs(sh_file_dir, exist_ok=True)
    
    for gpu_idx, command_chunk in zip(available_gpus, command_chunks):
        with open(f"{sh_file_dir}/run_{gpu_idx}.sh", "w") as f:
            f.write("\n".join([f'CUDA_VISIBLE_DEVICES={gpu_idx} {cmd}' for cmd in command_chunk]))
            
    print("Please run the following commands:")
    print("\n".join([f"nohup bash {sh_file_dir}/run_{gpu_idx}.sh > {sh_file_dir}/run_{gpu_idx}.log 2>&1 &" for gpu_idx in available_gpus]))
    
REGISTRY = {
    'local': local_launcher,
    'dummy': dummy_launcher,
    'multi_gpu': multi_gpu_launcher,
    "manual_multi_gpu": manual_multi_gpu_launcher
}

