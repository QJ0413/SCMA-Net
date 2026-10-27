"""
SCMA-Net entry point.
Usage:
  python scma_main.py --path ./checkpoints          # train
  python scma_main.py --path ./checkpoints --mode 2 # test
"""

import os
import cv2
import random
import numpy as np
import torch
import argparse
from shutil import copyfile

from src.config import Config
from src.SCMA import SCMA


def main(mode=None):
    config = load_config(mode)

    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:22'

    if torch.cuda.is_available():
        print('CUDA available')
        config.DEVICE = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        print('CUDA unavailable, using CPU')
        config.DEVICE = torch.device('cpu')

    cv2.setNumThreads(0)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    model = SCMA(config)
    model.load()

    if config.MODE == 1:
        config.print()
        print('\nstart training SCMA-Net...\n')
        model.train()
    elif config.MODE == 2:
        print('\nstart testing SCMA-Net...\n')
        model.test()


def load_config(mode=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '--checkpoints', type=str,
                        default='./checkpoints',
                        help='model checkpoints path (default: ./checkpoints)')
    parser.add_argument('--mode', type=int, default=1, choices=[1, 2],
                        help='1: train, 2: test')
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--mask',  type=str, default=None)
    parser.add_argument('--output', type=str, default=None)

    args = parser.parse_args()
    config_path = os.path.join(args.path, 'config.yml')

    if not os.path.exists(args.path):
        os.makedirs(args.path)
    if not os.path.exists(config_path):
        copyfile('./config.yml.example', config_path)

    config = Config(config_path)
    print(config_path)

    config.MODE = mode if mode is not None else args.mode

    if config.MODE == 2:
        config.MODEL = 2
        if args.input  is not None: config.TEST_INPAINT_IMAGE_FLIST = args.input
        if args.mask   is not None: config.TEST_MASK_FLIST           = args.mask
        if args.output is not None: config.RESULTS                   = args.output

    return config


if __name__ == '__main__':
    main()
