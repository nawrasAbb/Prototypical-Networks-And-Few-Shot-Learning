"""
Authors:
Nawras Abbas    315085043
Michael Bikman  317920999
Module with general utilities.
It is the 1st loaded module, so we put here all common objects.
"""

import logging
import os
import sys
from datetime import datetime

import torch

# current time
TIMESTAMP = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

# ---- Path to dataset location ---#
DATA_DIR = r"C:\DATASETS\mini-imagenet"
TRAIN_DATA_FILE = r"mini-imagenet-cache-train.pkl"
VALID_DATA_FILE = r"mini-imagenet-cache-val.pkl"
TEST_DATA_FILE = r"mini-imagenet-cache-test.pkl"

# device definition
DEVICE = torch.device('cpu')
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')


def beep(sec=1):
    """
    Will play a 1 second sound to signal the run ended
    """
    import winsound
    frequency = 440  # Set Frequency in Hertz
    duration = 1000 * sec  # Set Duration (in ms)
    winsound.Beep(frequency, duration)


def setup_reports():
    """
    Create setup for logger - both to file and to console
    Also model will be saved in the same folder with logs
    """
    report_root = 'reports'
    if not os.path.exists(report_root):
        os.makedirs(report_root)
    report_folder = f'{TIMESTAMP}_report'
    report_path = os.path.join(report_root, report_folder)
    if not os.path.exists(report_path):
        os.makedirs(report_path)

    logging.getLogger('matplotlib.font_manager').disabled = True
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    stdout_handler = logging.StreamHandler(sys.stdout)
    log_file = f'{TIMESTAMP}.log'
    output_file_handler = logging.FileHandler(f'{report_path}\{log_file}')
    logger.addHandler(output_file_handler)
    logger.addHandler(stdout_handler)
    return report_path
