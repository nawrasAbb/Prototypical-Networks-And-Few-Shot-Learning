"""
Authors:
Nawras Abbas    315085043
Michael Bikman  317920999
Module contains auxiliary parameter and result classes along with episode definition.
"""
import torch


class Episode(object):
    """
    Class representation for a single episode
    """

    def __init__(self):
        self.indexes = {}  # class_name -> (support pics indices, query pics indices)
        self.SUPPORT_DATA_NDX = 0
        self.QUERY_DATA_NDX = 1

    def get_support_sample(self, image_data):
        return self._get_data_sample(image_data, self.SUPPORT_DATA_NDX)

    def get_query_sample(self, image_data):
        return self._get_data_sample(image_data, self.QUERY_DATA_NDX)

    def add_indices(self, class_name, support_ndxs, query_ndxs):
        self.indexes[class_name] = (support_ndxs, query_ndxs)

    def _get_data_sample(self, image_data, data_index):
        all_samples = []
        for class_name in self.indexes.keys():
            ndxs = self.indexes[class_name][data_index]
            sample_indices = torch.tensor(ndxs)
            sample = torch.index_select(image_data, 0, sample_indices)
            sample = sample.transpose(1, 3).transpose(2, 3)
            all_samples.append(sample)
        result = torch.stack(all_samples, dim=0)
        return result


class RunParameters(object):
    """
    Use this class for train + validation run
    """

    def __init__(self):
        self.model = None
        self.epochs = 0
        self.loss = None
        self.optimizer = None
        self.scheduler = None
        self.report_path = None  # path where to save PT file and logs
        self.train_data = None
        self.val_data = None
        self.episodes_per_epoch = 0
        self.patience = 0
        self.n_ways = 0
        self.n_support_examples = 0
        self.n_query_examples = 0

    def __str__(self):
        return '--------Run params--------\n' \
               f'Epochs: {self.epochs}\n' \
               f'Loss: {self.loss}\n' \
               f'Optimizer:{self.optimizer}\n' \
               f'Scheduler:{self.scheduler}\n' \
               f'report_path:{self.report_path}\n' \
               f'episodes_per_epoch:{self.episodes_per_epoch}\n' \
               f'patience:{self.patience}\n' \
               f'n_ways:{self.n_ways}\n' \
               f'n_support_examples:{self.n_support_examples}\n' \
               f'n_query_examples:{self.n_query_examples}\n' \
               '-----------------------------\n'


class TrainResult(object):
    """
    Result of train + validation run
    """

    def __init__(self):
        self.train_loss_per_epoch = []
        self.train_accuracy_per_epoch = []
        self.validation_loss_per_epoch = []
        self.validation_accuracy_per_epoch = []
        self.best_epoch = 0

    def train_loss_min(self):
        return min(self.train_loss_per_epoch)

    def valid_loss_min(self):
        return min(self.validation_loss_per_epoch)

    def train_accuracy(self):
        return max(self.train_accuracy_per_epoch)

    def validation_accuracy(self):
        return max(self.validation_accuracy_per_epoch)

    def __str__(self):
        return '--------Run result--------\n' \
               f'Train Loss: {self.train_loss_min()}\n' \
               f'Train Acc: {self.train_accuracy()}\n' \
               f'Valid Loss:{self.valid_loss_min()}\n' \
               f'Valid Acc:{self.validation_accuracy()}\n' \
               f'Best Epoch:{self.best_epoch}\n' \
               '-----------------------------\n'


class TestResult:
    """
    Used for test evaluation
    """

    def __init__(self):
        self.acc = 0  # accuracy
        self.loss = 0  # loss

    def __str__(self):
        return '--------Test result--------\n' \
               f'Test Acc: {self.acc}\n' \
               f'Test Loss:{self.loss}\n' \
               '-----------------------------\n'
