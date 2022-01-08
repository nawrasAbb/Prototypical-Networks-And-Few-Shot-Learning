"""
Authors:
Nawras Abbas    315085043
Michael Bikman  317920999
This module allows to run test evaluation on a selected model
"""
import logging
import os
import pickle
import random

import torch

from engine import test
from models import *
from parameters import RunParameters
from utils import beep, setup_reports, DATA_DIR, TEST_DATA_FILE, DEVICE

# select seed to get consistent results across different models
SEED = 1111
# put True to use seeded random
USE_SEED = True
# put True to enable logging
LOG_ENABLED = True
# select model file name
MODEL_FILE = 'model.pt'  # must be in the same directory as code

log = print
if LOG_ENABLED:
    log = logging.info


def load_test_data():
    """
    Load test evaluation data from Pickle file
    :return: data dictionary
    """
    with open(os.path.join(DATA_DIR, TEST_DATA_FILE), "rb") as data_file:
        data = pickle.load(data_file)
    log(f'Loaded test data: {TEST_DATA_FILE}')
    return data


def main():
    """
    Runs evaluation
    """
    setup_reports()
    log(DEVICE)
    log(f'SEED:{SEED}')

    # --- Select model here ---#
    # model = ProtoNetSimple().to(DEVICE)
    # model = ResNetSimple().to(DEVICE)
    # model = ProtoNetRes18().to(DEVICE)
    model = ProtoNetRes12().to(DEVICE)

    log(model)
    # model.alpha = None #uncomment to disable scaled distance

    if not os.path.isfile(MODEL_FILE):
        raise Exception(f'Cannot train model - no saved state found: {MODEL_FILE}!')
    state_dict = torch.load(MODEL_FILE)
    model.load_state_dict(state_dict)

    log(f'Alpha:{list(model.parameters())[0].item()}')

    log('Model load successful!')
    # load model state
    test_data = load_test_data()
    # ran training
    test_params = RunParameters()
    test_params.model = model
    test_params.val_data = test_data
    # we test 5-class-5-shot
    test_params.n_ways = 5  # ways
    test_params.episodes_per_epoch = 100
    test_params.n_support_examples = 5  # shots
    test_params.n_query_examples = 15
    log(test_params)

    if USE_SEED:
        random.seed(SEED)

    test_res = test(test_params, log)
    log(test_res)


if __name__ == '__main__':
    main()
    print('OK')
    beep()  # Play sound alarm
