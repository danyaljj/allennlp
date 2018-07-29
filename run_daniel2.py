#!/usr/bin/env python
import logging
import os
import sys

import torch
from torch import nn

import torch.nn.functional as F

from allennlp.common.util import ensure_list
from allennlp.data.dataset_readers import SquadReader
from allennlp.models.archival import load_archive, Archive
from allennlp.predictors import Predictor
from allennlp.data import DatasetReader
from allennlp.data.dataset import Batch

if __name__ == "__main__":

    a = torch.tensor([[1., -1.], [1., -1.]])
    print(a)

    print(a.data.numpy())

    print(len(a.data.numpy()))

    print(a.data.numpy().flatten())

