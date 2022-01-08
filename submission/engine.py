"""
Authors:
Nawras Abbas    315085043
Michael Bikman  317920999
This is the main train, test and validation logic module.
It creates the episodes, runs training, testing and validation flow
"""
import math
import random
import time

import torch
from tqdm import tqdm

from parameters import TrainResult, Episode, TestResult


def choose_episode_classes(class_dict, n_ways):
    """
    Choose which classes will participate in the episode
    :param class_dict: dictionary of all classes
    :param n_ways: number of classes per episode
    :return: classes for the episode
    """
    # select classes for episode (random uniform)
    all_classes = class_dict.keys()
    classes = random.sample(all_classes, n_ways)
    return classes


def create_episodes(class_dict, n_episodes, n_ways, n_supports, n_queries):
    """
    Returns list of episode objects
    :param class_dict: parsed from dataset
    :param n_episodes: number of episodes to return
    :param n_ways: number of ways per episode
    :param n_supports: number of support examples per episodes
    :param n_queries: number of query examples per episode
    """
    episodes = []

    for e in range(n_episodes):
        try:
            classes = choose_episode_classes(class_dict, n_ways)
        except ValueError:
            continue

        episode = Episode()
        for c in classes:
            class_ndxs = class_dict[c]
            selected_ndxs = random.sample(class_ndxs, n_supports + n_queries)

            support_ndxs = selected_ndxs[:n_supports]
            query_ndxs = selected_ndxs[n_supports:]
            episode.add_indices(c, support_ndxs, query_ndxs)

        episodes.append(episode)

    return episodes


def train(parameters, log):
    """
    Run training procedure
    :param log: logger to file / console
    :param parameters: training parameters context
    :return: train result
    """
    log('Train started')
    train_res = TrainResult()

    best_loss = math.inf
    epochs_since_best = 0
    for ep in range(parameters.epochs):
        epoch_start_time = time.time()

        train_loop(ep, parameters, train_res, log)
        validation_loop(parameters, train_res, log, f"validation")
        if parameters.scheduler is not None:
            parameters.scheduler.step()

        # early stopping
        epoch_loss = train_res.validation_loss_per_epoch[-1].item()
        if epoch_loss < best_loss:
            epochs_since_best = 0
            best_loss = epoch_loss
            parameters.model.save(parameters.report_path)
            log('best model saved')
        else:
            epochs_since_best = epochs_since_best + 1
        if epochs_since_best > parameters.patience:
            # update result and get out
            train_res.best_epoch = ep - parameters.patience
            return train_res

        elapsed_time = time.time() - epoch_start_time
        log(f'time: {elapsed_time:5.2f} sec')

    return train_res


def train_loop(ep, parameters, train_res, log):
    """
    Runs training and fills result
    :param ep: episode number
    :param parameters: train parameters
    :param train_res: train result
    :param log: logger
    """
    model = parameters.model
    optimizer = parameters.optimizer
    n_ways = parameters.n_ways
    n_episodes = parameters.episodes_per_epoch
    n_supports = parameters.n_support_examples
    n_queries = parameters.n_query_examples
    train_image_data = torch.from_numpy(parameters.train_data['image_data'])
    train_class_dict = parameters.train_data['class_dict']

    model.train()
    train_batch_losses = []
    train_batch_accuracies = []
    train_episodes = create_episodes(train_class_dict, n_episodes, n_ways, n_supports, n_queries)
    log(f'Epoch {ep + 1}')
    for episode in tqdm(train_episodes, desc=f"Epoch {ep + 1}/{parameters.epochs} train"):
        optimizer.zero_grad()
        loss, output = model.loss(episode, train_image_data)
        train_batch_losses.append(output['loss'])
        train_batch_accuracies.append(output['acc'])
        loss.backward()
        optimizer.step()
    epoch_train_loss = torch.mean(torch.tensor(train_batch_losses))
    epoch_train_acc = torch.mean(torch.tensor(train_batch_accuracies))
    log(f'train loss: {epoch_train_loss:.6f}')
    log(f'train acc: {epoch_train_acc:.6f}')
    # update train losses in result here
    train_res.train_accuracy_per_epoch.append(epoch_train_acc)
    train_res.train_loss_per_epoch.append(epoch_train_loss)


def validation_loop(parameters, train_res, log, desc):
    """
    Runs validation and fills result
    :param desc: description string
    :param parameters: train parameters
    :param train_res: train results
    :param log: logger
    """
    model = parameters.model
    n_ways = parameters.n_ways
    n_episodes = parameters.episodes_per_epoch
    n_supports = parameters.n_support_examples
    n_queries = parameters.n_query_examples
    val_image_data = torch.from_numpy(parameters.val_data['image_data'])
    val_class_dict = parameters.val_data['class_dict']

    model.eval()
    val_batch_losses = []
    val_batch_accuracies = []
    with torch.no_grad():
        val_episodes = create_episodes(val_class_dict, n_episodes, n_ways, n_supports, n_queries)
        for episode in tqdm(val_episodes, desc=desc):
            _, output = model.loss(episode, val_image_data)
            val_batch_losses.append(output['loss'])
            val_batch_accuracies.append(output['acc'])
    epoch_val_loss = torch.mean(torch.tensor(val_batch_losses))
    epoch_val_acc = torch.mean(torch.tensor(val_batch_accuracies))
    log(f'{desc} loss: {epoch_val_loss:.6f}')
    log(f'{desc} acc: {epoch_val_acc:.6f}')
    # update losses and accuracies in result here
    train_res.validation_accuracy_per_epoch.append(epoch_val_acc)
    train_res.validation_loss_per_epoch.append(epoch_val_loss)


def test(parameters, log):
    """
    Run test evaluation
    :param parameters: test parameters
    :param log: logger
    :return: test result
    """
    log('Test started')
    test_res = TrainResult()
    epoch_start_time = time.time()
    validation_loop(parameters, test_res, log, f"Test")
    elapsed_time = time.time() - epoch_start_time
    log(f'time: {elapsed_time:5.2f} sec')
    result = TestResult()
    result.acc = test_res.validation_accuracy_per_epoch[-1].item()
    result.loss = test_res.validation_loss_per_epoch[-1].item()
    return result


if __name__ == '__main__':
    # import pickle
    # with open(r'C:\DATASETS\mini-imagenet\mini-imagenet-cache-train.pkl', "rb") as data_file:
    #     data = pickle.load(data_file)
    # res = create_episodes(data['class_dict'], 100, 20, 15, 5)
    pass
