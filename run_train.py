"""
Authors:
Nawras Abbas    315085043
Michael Bikman  317920999
This module allows to run training on a selected model
"""
import logging
import os
import pickle

import torch

from engine import train
from models import *
from parameters import RunParameters
from utils import beep, setup_reports, DATA_DIR, TRAIN_DATA_FILE, VALID_DATA_FILE, DEVICE

# put True to enable logging
LOG_ENABLED = True

log = print
if LOG_ENABLED:
    log = logging.info


def load_train_data():
    """
    Load train split data from Pickle file
    :return: data dictionary
    """
    with open(os.path.join(DATA_DIR, TRAIN_DATA_FILE), "rb") as data_file:
        data = pickle.load(data_file)
    log(f'Loaded train data: {TRAIN_DATA_FILE}')
    return data


def load_validation_data():
    """
    Load validation split data from Pickle file
    :return: data dictionary
    """
    with open(os.path.join(DATA_DIR, VALID_DATA_FILE), "rb") as data_file:
        data = pickle.load(data_file)
    log(f'Loaded validation data: {VALID_DATA_FILE}')
    return data


def main():
    """
    Runs training
    """
    report_path = setup_reports()
    log(DEVICE)

    num_filters = 128
    log(f'Num Filters: {num_filters}')
    drop_rate = 0.1
    log(f'Drop block rate: {drop_rate}')

    # --- Select model here ---#
    # model = ProtoNetSimple(num_filters).to(DEVICE)
    # model = ProtoNetComplex().to(DEVICE) # doesnt learn!
    # model = ResNetSimple(num_filters).to(DEVICE)
    # model = ProtoNetRes18().to(DEVICE)
    model = ProtoNetRes12(drop_rate).to(DEVICE)

    log(model)
    log(f'Alpha:{list(model.parameters())[0].item()}')

    # select optimization parameters
    lr = 0.002
    log(f'LR:{lr}')
    step = 50
    log(f'LR Step:{step}')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=0.5, last_epoch=-1)
    # scheduler = None

    # load train & validation data
    train_data = load_train_data()
    val_data = load_validation_data()
    train_params = RunParameters()
    train_params.epochs = 10000  # max number of epochs
    train_params.optimizer = optimizer
    train_params.scheduler = scheduler
    train_params.train_data = train_data
    train_params.val_data = val_data
    # NOTE: do not use too much ways per episode - GPU fails out of memory :(
    train_params.n_ways = 5  # ways
    train_params.patience = 200
    train_params.episodes_per_epoch = 100
    train_params.n_support_examples = 5  # shots
    train_params.n_query_examples = 15
    train_params.model = model
    train_params.report_path = report_path
    log(train_params)
    train_res = train(train_params, log)
    log(train_res)


if __name__ == '__main__':
    main()
    print('OK')
    beep()  # Play sound alarm
