import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import torch
from torchtext.data import BucketIterator
from torchtext.datasets import WMT14, Multi30k, TranslationDataset
from utils import arg_parser
from utils.vocab import SRC, TRG


def get_args():
    parser = arg_parser.create_parser()
    args = parser.parse_args()
    return args


def get_data(dataset):
    """ 
    Loads train, validation and test dataset for training and evaluation.
    ------------------------------------
    dataset (str): Choice of dataset
    ------------------------------------
    Returns train, validation and test dataset
    """
    # TODO: add more data
    if dataset == "multi30k":
        # WMT16 Multimodal Dataset, see https://www.statmt.org/wmt16/multimodal-task.html
        train_data, valid_data, test_data = Multi30k.splits(exts = ('.en', '.de'), 
                                                            fields = (SRC, TRG))
    elif dataset == 'wmt14':
        # TODO: Check if implementation of WMT14 data loading is correct
        # train_data = TranslationDataset(path='.data/wmt14/train', exts=('.en', '.de'), fields=(SRC, TRG)) # takes 10min to execute
        train_data = TranslationDataset(path='.data/wmt14/newstest2015', exts=('.en', '.de'), fields=(SRC, TRG)) # for debugging
        valid_data = TranslationDataset(path='.data/wmt14/newstest2015', exts=('.en', '.de'), fields=(SRC, TRG)) 
        test_data = TranslationDataset(path='.data/wmt14/newstest2015', exts=('.en', '.de'), fields=(SRC, TRG)) # for debugging
    elif dataset == 'testwmt14':
        # TODO: Check if WMT14 class implemetation is correct
        # train_data = WMT14(path='.data/wmt14/train', exts=('.en', '.de'), fields=(SRC, TRG)) # takes 10min to execute
        train_data = WMT14(path='.data/wmt14/newstest2015', exts=('.en', '.de'), fields=(SRC, TRG)) # for debugging
        valid_data = WMT14(path='.data/wmt14/newstest2015', exts=('.en', '.de'), fields=(SRC, TRG))
        test_data = WMT14(path='.data/wmt14/newstest2015', exts=('.en', '.de'), fields=(SRC, TRG)) # for debugging
    return train_data, valid_data, test_data

def get_dataloader(train_data, valid_data, test_data, batch_size, device):
    """
    Converts train/valid/test data into dataloader.
    ------------------------------------------------
    train_data (torchtext.data): Training Dataset
    valid_data (torchtext.data): Validation Dataset
    test_data (torchtext.data): Test Dataset
    batch_size (int): Batch Size
    device (str): CUDA or CPU
    ------------------------------------------------
    Returns train, validation and test dataloader
    """
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data), 
        batch_size = batch_size,
        device = device)
    return train_iterator, valid_iterator, test_iterator
