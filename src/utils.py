import torch
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

import os
import sys
import logging

from src.argument import BasicArguments, DataTrainingArguments, ModelArguments, TrainingArguments


def get_basic_parser():
    parser = BasicArguments()
    args = parser.parse()

    # select GPU if available
    use_cuda = torch.cuda.is_available()
    args.device = 'cuda:0' if use_cuda else 'cpu'

    return args

def get_data_parser():
    parser = DataTrainingArguments()
    args = parser.parse()

    return args

def get_model_parser():
    parser = ModelArguments()
    args = parser.parse()

    return args

def get_training_parser():
    parser = TrainingArguments()
    args = parser.parse()

    return args

def get_logger(is_main=True, filename=None): 
    """
    Get logger to store information about the script and track events that occur

    :param
        is_main: bool
                true if training 
        filename: str
                name of log file
    
    :return
        loggger: logging
                log infomation
    """
    logger = logging.getLogger(__name__)

    handlers = [logging.StreamHandler(sys.stdout)]
    if filename is not None: 
        handlers.append(logging.FileHandler(filename=filename))
    logging.basicConfig(
        datefmt="%m/%d/%Y %H:%M:%S", 
        level=logging.INFO if is_main else logging.WARN, 
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s", 
        handlers=handlers
    )       
    return logger


def log_gpu_utilization(logger):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    logger.info(f"GPU memory occupied: {info.used//1024**2} MB.")

def init_checkpoint_folder(basic_args):
    """
    Init checkpoint path and make checkpoint directory

    :param
        args: Arguments
                contains hyper-parameter
    """
    basic_args.checkpoint_path = os.path.join(basic_args.checkpoint_dir, basic_args.name)
    basic_args.checkpoint_exists = os.path.exists(basic_args.checkpoint_path)
    if not basic_args.checkpoint_exists:
        os.makedirs(basic_args.checkpoint_path)
