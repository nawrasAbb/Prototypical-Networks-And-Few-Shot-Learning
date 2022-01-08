"""
Authors:
Nawras Abbas    315085043
Michael Bikman  317920999
This module contains all the mode;s (more and less successfully) used in our project
All the models inherit from the first one to share the same loss function!
"""
import os

import numpy as np
import torch.nn
from torch import nn
from torch.autograd import Variable
from torch.nn import functional

from res_net_12 import resnet12
from res_net_18 import ResNet18
from utils import DEVICE


def euclidean_dist(x, y):
    """
    Calculates Euclidean distance (square of it)
    :param x: size [n_query_total, out_dim=1600] - queries
    :param y: size [n_ways, out_dim=1600] - prototypes
    """
    n = x.size(0)  # total number of query points = n_query_total
    m = y.size(0)  # number of classes = n_ways
    d = x.size(1)  # dimension of pic embedding = 1600 for mini-ImageNet
    if d != y.size(1):
        raise ValueError(f'Pic embedding for prototype {y.size(1)} and query {d} data arent equal')

    x = x.unsqueeze(1).expand(n, m, d)  # size = [n_query_total, n_ways, 1600]
    y = y.unsqueeze(0).expand(n, m, d)  # size = [n_query_total, n_ways, 1600]

    return torch.pow(x - y, 2).sum(2)


def mahalanobis_dist(x, y):
    """
    Calculates Mahalanobis (generalized) distance
    :param x: size [n_query_total, out_dim=1600] - queries
    :param y: size [n_ways, out_dim=1600] - prototypes
    """
    n_queries = 15
    n_query_total = x.size(0)
    n_ways = y.size(0)  # number of classes = n_ways
    res = torch.zeros(n_query_total, n_ways).to(DEVICE)  # size = [n_query_total, n_ways]
    queries_per_class = x.split(n_queries, dim=0)  # (10, [15, 1600])
    prototypes_per_class = y.split(1, dim=0)  # (10, [1,1600])
    batches = int(n_query_total / n_queries)
    for class_ndx in range(n_ways):
        # print(f'class_ndx={class_ndx}')
        class_queries = queries_per_class[class_ndx].detach().cpu()
        proto = prototypes_per_class[class_ndx]
        for query_batch_ndx in range(batches):
            query_batch = queries_per_class[query_batch_ndx]
            cov_arr = np.cov(class_queries.T)
            cov = torch.from_numpy(cov_arr).to(DEVICE)
            cov_diag = torch.diag(cov)
            cov = torch.diag(cov_diag)

            # print(f'query_batch={query_batch_ndx}')
            for query_ndx in range(n_queries):
                # print(f'query_ndx={query_ndx}')
                query = query_batch[query_ndx, :]
                dist = mahalanobis(proto, query, cov)
                # print(dist.item())
                q = n_queries * query_batch_ndx + query_ndx
                res[q, class_ndx] = dist.item()
    return res


def mahalanobis(u, v, cov):
    """
    Calculates mahalanobis distance between 2 vectors
    :param u: vector
    :param v: vector
    :param cov: covariance matrix
    :return:
    """
    delta = (u - v).double()
    delta_trans = torch.transpose(delta, 0, 1).double()
    cov_inverse = torch.inverse(cov).double()
    mult = torch.matmul(delta, cov_inverse)
    m = torch.matmul(mult, delta_trans)
    return torch.sqrt(m)


class ProtoNetSimple(nn.Module):
    """
    Model 1 – CNN with constant number of feature maps per layer
    """
    def __init__(self, num_filters=128):
        super().__init__()

        self.file_name = 'ProtoNetSimple.pt'

        self.block1 = self._cnn_block(3, num_filters)
        self.block2 = self._cnn_block(num_filters, num_filters)
        self.block3 = self._cnn_block(num_filters, num_filters)
        self.block4 = self._cnn_block(num_filters, num_filters)

        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1)
        )

        self.alpha = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        # self.alpha = None

    @staticmethod
    def _cnn_block(in_channels, out_channels):
        block = torch.nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(0.2),
            nn.MaxPool2d(kernel_size=2)
        )
        return block

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        vector = self.classifier(x)
        return vector

    def loss(self, episode, image_data):
        """
        Custom loss function to calculate prototype and distance to it
        :param episode: episode object
        :param image_data: tensor for image
        :return: loss value, dictionary of
        """
        support_data = episode.get_support_sample(image_data)
        query_data = episode.get_query_sample(image_data)

        xs = Variable(support_data)
        # xs size = [n_ways, n_shots, channels=3, width=84, height=84]
        xq = Variable(query_data)
        # xq size = [n_ways, n_query_points, channels=3, width=84, height=84]

        n_class = xs.size(0)
        if xq.size(0) != n_class:
            raise ValueError(f'Number of classes for support {xs.size(0)} and query {xq.size(0)} data is not equal')
        n_support = xs.size(1)
        n_query = xq.size(1)

        target_indices = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long().to(DEVICE)
        target_indices = Variable(target_indices, requires_grad=False)

        support_pic_size = xs.size()[2:]
        query_pic_size = xq.size()[2:]
        if query_pic_size != support_pic_size:
            raise ValueError(f'Pic sizes for support {support_pic_size} and query {query_pic_size} data arent equal')
        n_support_total = n_class * n_support
        n_query_total = n_class * n_query
        xs_view = xs.view(n_support_total, *support_pic_size)
        xq_view = xq.view(n_query_total, *query_pic_size)
        x = torch.cat([xs_view, xq_view], 0).float().to(DEVICE)  # input for the model
        # x = has dimension of [n_support_total + n_query_total, channels=3, width=84, height=84]

        z = self.forward(x)  # output with dimension [n_support_total + n_query_total, 1600]
        z_dim = z.size(-1)

        z_support = z[:n_support_total]  # size = [n_support_total, 1600]
        # prototype = average all embeddings from support set
        prototypes_per_class = z_support.view(n_class, n_support, z_dim).mean(1)  # size = [n_class, 1600]
        query_vectors = z[n_support_total:]  # size = [n_query_total, 1600]

        dists_per_class = euclidean_dist(query_vectors, prototypes_per_class)  # size = [n_query_total, n_ways]
        # dists_per_class = mahalanobis_dist(query_vectors, prototypes_per_class)  # size = [n_query_total, n_ways]

        # alpha used here ----------------------------------------------------------
        # alpha parameter is used to scale the distance metric to obtain better results
        if self.alpha is not None:
            dists_per_class = torch.mul(self.alpha, dists_per_class)
        # --------------------------------------------------------------------------

        log_p_y = torch.nn.functional.log_softmax(-dists_per_class, dim=1).view(n_class, n_query, -1)  # log(p(y=k|x))
        # log_p_y = size [n_class = n_ways, n_query, n_class]
        loss_per_query = -log_p_y.gather(2, target_indices).squeeze().view(-1)  # size = [n_query_total]
        loss_val = loss_per_query.mean()  # average loss for all queries
        _, y_hat = log_p_y.max(dim=2)  # returns tuple (max values, argmax indices)
        # y_hat size = [n_class, n_query]

        # calculate accuracy = number of matches between y_hat indices and ground truth target_indices
        acc_val = torch.eq(y_hat, target_indices.squeeze()).float().mean()

        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item()
        }

    def save(self, report_folder):
        model_file = os.path.join(report_folder, self.file_name)
        torch.save(self.state_dict(), model_file)


class ProtoNetComplex(ProtoNetSimple):
    """
    This model is very bad and doesnt learn!
    """

    def __init__(self, num_filters=64):
        super().__init__(num_filters)

        self.file_name = 'ProtoNetComplex.pt'

        self.block1 = self._cnn_block(3, num_filters)
        self.block2 = self._cnn_block(num_filters, num_filters)
        self.block3 = self._cnn_block(num_filters, num_filters)
        self.block4 = self._cnn_block(num_filters, num_filters)

        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(1600, 1600),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1600, 128)
        )


class ProtoNetRes18(ProtoNetSimple):
    """
    Model 3 -	Residual deep network based on ResNet18
    """
    def __init__(self):
        super().__init__()
        self.res_net = ResNet18()
        self.file_name = 'ProtoNetRes18.pt'

    def forward(self, x):
        x = self.res_net.forward(x)
        return x


class ProtoNetRes12(ProtoNetSimple):
    """
    Model 4 - Residual network with Drop Blocks
    """
    def __init__(self, drop_rate=0.1):
        super().__init__()
        self.res_net = resnet12(drop_rate=drop_rate)
        self.file_name = 'ProtoNetRes12.pt'

    def forward(self, x):
        x = self.res_net.forward(x)
        return x


class ResNetSimple(ProtoNetSimple):
    """
    Model 2 - Densely Connected CNN
    """

    def __init__(self, num_filters=64):
        super().__init__()
        self.file_name = 'ResNetSimple.pt'
        self.maxPool = nn.MaxPool2d(kernel_size=2)
        self.alpha = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Sequential(
            nn.Flatten(start_dim=1)
        )
        self.BN = nn.BatchNorm2d(num_filters)

        self.block1 = self._cnn_block(3, num_filters)
        self.blockRes1 = self._cnn_block_Res(3, num_filters)

        self.block2 = self._cnn_block(num_filters, num_filters)
        self.blockRes2 = self._cnn_block_Res(num_filters, num_filters)

    @staticmethod
    def _cnn_block(in_channels, out_channels):
        block = torch.nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        return block

    @staticmethod
    def _cnn_block_Res(in_channels, out_channels):
        block = torch.nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        return block

    def forward(self, x):
        # **************************   block 1   *********************************
        # on the original input of the block we run a conv layer and Batch normalization
        residual = x
        residual = self.blockRes1(residual)

        # on the original input run conv -> relu -> conv -> relu -> conv
        # and then sum the outputs of the above two blocks and then run relu and MP
        # run the same architecture on the other 3 blocks
        x = self.block1(x)
        x = x + residual
        x = self.relu(x)
        x = self.maxPool(x)

        # **************************   block 2   *********************************
        residual = x
        residual = self.blockRes2(residual)
        x = self.block2(x)
        x = x + residual
        x = self.relu(x)
        x = self.maxPool(x)

        # **************************   block 3   *********************************
        residual = x
        residual = self.blockRes2(residual)
        x = self.block2(x)
        x = x + residual
        x = self.relu(x)
        x = self.maxPool(x)

        # **************************   block 4   *********************************
        residual = x
        residual = self.blockRes2(residual)
        x = self.block2(x)
        x = x + residual
        x = self.relu(x)
        x = self.maxPool(x)

        vector = self.flatten(x)
        return vector


if __name__ == '__main__':
    # x = torch.randint(100, (150, 20)).to(DEVICE)
    # y = torch.randint(100, (10, 20)).to(DEVICE)
    # dist = mahalanobis_dist(x, y)
    # arr_res = dist.detach().cpu().numpy()
    # dist2 = euclidean_dist(x, y)
    # arr_res2 = dist2.detach().cpu().numpy()

    from torchsummary import summary

    # model = ProtoNetSimple(128).to(DEVICE)
    # model = ResNetSimple(64).to(DEVICE)
    # model = ProtoNetRes18().to(DEVICE)
    model = ProtoNetRes12().to(DEVICE)
    summary(model, (3, 84, 84))

    pass
