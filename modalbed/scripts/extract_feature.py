import argparse
import os
import random
import sys
import time
from tqdm import tqdm

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data

from modalbed import datasets
from modalbed import hparams_registry
from modalbed import algorithms
from modalbed.lib import misc
from modalbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str, default="MSR_VTT")
    parser.add_argument('--perceptor', type=str, default="imagebind")
    parser.add_argument('--output_dir', type=str, default="extract_feature_output")
    args = parser.parse_args()

    start_step = 0
    algorithm_dict = None

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    hparams = hparams_registry.default_hparams( args.perceptor, "ERM", args.dataset)
    
    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
            [0], hparams)
    else:
        raise NotImplementedError

    in_splits = []
    for env_i, env in enumerate(dataset):
        uda = []
        _, in_ = misc.split_dataset(env,
            0,
            misc.seed_hash(0, env_i))
        in_splits.append((in_, None))

    data_loader = [FastDataLoader(
        dataset=env,
        # weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)]
    
    from ..perceptors import ImagebindPreceptor, LanguageBindPreceptor, UniBindPreceptor, StrongMGPreceptor
    
    if args.perceptor == "imagebind":
        perceptor = ImagebindPreceptor(args.dataset, True)
    elif args.perceptor == "languagebind":
        perceptor = LanguageBindPreceptor(args.dataset, True)
    elif args.perceptor == "unibind":
        perceptor = UniBindPreceptor(args.dataset, True)
    elif args.perceptor == "strongMG":
        perceptor = StrongMGPreceptor(args.dataset, True)
    else:
        raise NotImplementedError
    perceptor.to(device)

    minibatches_iterator = zip(*data_loader)

    steps_per_epoch = min([len(env)/hparams['batch_size'] for env,_ in in_splits])

    n_steps = dataset.N_STEPS
    checkpoint_freq = dataset.CHECKPOINT_FREQ

    last_results_keys = None
    for step in tqdm(range(start_step, n_steps)):
        step_start_time = time.time()
        minibatches_device = [(x, y)
            for x,y in next(minibatches_iterator)]
        uda_device = None
        with torch.no_grad():
            step_vals = perceptor.update(minibatches_device, uda_device)

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')
