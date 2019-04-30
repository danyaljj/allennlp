import json
import logging
import os
from collections import Iterable
from typing import Dict, Iterable

import sys

import numpy
import numpy as np
import re
from sklearn.metrics import confusion_matrix
from scipy import linalg
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier

from allennlp.commands import Train
from allennlp.common import Params
from allennlp.data import DataIterator
from allennlp.data import Instance
from allennlp.data import Vocabulary
from allennlp.data.dataset_readers import SquadReader
from allennlp.models import BidirectionalAttentionFlow
from allennlp.models import Model
from allennlp.predictors import Predictor
from allennlp.data import DatasetReader
from allennlp.data.dataset import Batch
from sklearn import cluster, metrics
import time as time
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets.samples_generator import make_swiss_roll
from scipy.cluster.hierarchy import dendrogram, linkage

from allennlp.training import Trainer


def load_model():
    from allennlp.models import load_archive
    archive = load_archive("https://s3-us-west-2.amazonaws.com/allennlp/models/bidaf-model-2017.09.15-charpad.tar.gz")
    # archive = load_archive("finetune_factor_001_epoch_2/model.tar.gz")
    config = archive.config.duplicate()
    model = archive.model
    model.eval()
    dataset_reader_params = config["dataset_reader"]
    dataset_reader = DatasetReader.from_params(dataset_reader_params)

    return model, dataset_reader


def datasets_from_params(params: Params) -> Dict[str, Iterable[Instance]]:
    """
    Load all the datasets specified by the config.
    """
    dataset_reader = DatasetReader.from_params(params.pop('dataset_reader'))
    validation_dataset_reader_params = params.pop("validation_dataset_reader", None)

    validation_and_test_dataset_reader: DatasetReader = dataset_reader
    if validation_dataset_reader_params is not None:
        # logger.info("Using a separate dataset reader to load validation and test data.")
        validation_and_test_dataset_reader = DatasetReader.from_params(validation_dataset_reader_params)

    train_data_path = params.pop('train_data_path')
    # logger.info("Reading training data from %s", train_data_path)
    train_data = dataset_reader.read(train_data_path)

    datasets: Dict[str, Iterable[Instance]] = {"train": train_data}

    validation_data_path = params.pop('validation_data_path', None)
    if validation_data_path is not None:
        # logger.info("Reading validation data from %s", validation_data_path)
        validation_data = validation_and_test_dataset_reader.read(validation_data_path)
        datasets["validation"] = validation_data

    test_data_path = params.pop("test_data_path", None)
    if test_data_path is not None:
        # logger.info("Reading test data from %s", test_data_path)
        test_data = validation_and_test_dataset_reader.read(test_data_path)
        datasets["test"] = test_data

    return datasets


def train():
    params = Params.from_file("/Users/daniel/ideaProjects/allennlp/allennlp/knn/configs/config_zero_modeling_local.json")

    all_datasets = datasets_from_params(params)
    datasets_for_vocab_creation = set(params.pop("datasets_for_vocab_creation", all_datasets))
    vocab = Vocabulary.from_params(
        params.pop("vocabulary", {}),
        (instance for key, dataset in all_datasets.items()
         for instance in dataset
         if key in datasets_for_vocab_creation)
    )

    model = Model.from_params(vocab=vocab, params=params.pop('model'))
    # reader = SquadReader()

    # vocab.save_to_files(os.path.join(serialization_dir, "vocabulary"))

    iterator = DataIterator.from_params(params.pop("iterator"))
    iterator.index_with(vocab)
    validation_iterator_params = params.pop("validation_iterator", None)
    if validation_iterator_params:
        validation_iterator = DataIterator.from_params(validation_iterator_params)
        validation_iterator.index_with(vocab)
    else:
        validation_iterator = None

    train_data = all_datasets['train']
    validation_data = all_datasets.get('validation')
    test_data = all_datasets.get('test')

    trainer_params = params.pop("trainer")
    no_grad_regexes = trainer_params.pop("no_grad", ())
    for name, parameter in model.named_parameters():
        if any(re.search(regex, name) for regex in no_grad_regexes):
            parameter.requires_grad_(False)

    # frozen_parameter_names, tunable_parameter_names = get_frozen_and_tunable_parameter_names(model)
    # logger.info("Following parameters are Frozen  (without gradient):")
    # for name in frozen_parameter_names:
    #     logger.info(name)
    # logger.info("Following parameters are Tunable (with gradient):")
    # for name in tunable_parameter_names:
    #     logger.info(name)

    trainer_choice = trainer_params.pop_choice("type",
                                               Trainer.list_available(),
                                               default_to_first_choice=True)
    trainer = Trainer.by_name(trainer_choice).from_params(model=model,
                                                          serialization_dir="out3",
                                                          iterator=iterator,
                                                          train_data=train_data,
                                                          validation_data=validation_data,
                                                          params=trainer_params,
                                                          validation_iterator=validation_iterator)

    metrics = trainer.train()




def load_model():
    from allennlp.models import load_archive
    archive = load_archive("https://s3-us-west-2.amazonaws.com/allennlp/models/bidaf-model-2017.09.15-charpad.tar.gz")
    # archive = load_archive("finetune_factor_001_epoch_2/model.tar.gz")
    config = archive.config.duplicate()
    model = archive.model
    model.eval()
    dataset_reader_params = config["dataset_reader"]
    dataset_reader = DatasetReader.from_params(dataset_reader_params)

    return model, dataset_reader

def create_instance(reader, question, paragraph, answer_start, answer_end):
    return reader.text_to_instance(question, paragraph, zip([answer_start], [answer_end]))

def solve_sample_question():
    # model, dataset_reader = load_model()
    model, dataset_reader = train()

    instance1 = create_instance(reader=dataset_reader, question = "What kind of test succeeded on its first attempt?",
                    paragraph="One time I was writing a unit test, and it succeeded on the first attempt.",
                                answer_start=1, answer_end=5)

    instance2 = create_instance(reader=dataset_reader, question="What happens when you go cheap?",
                                paragraph="Dion Lewis on Blowout Win vs. Patriots: 'That's What Happens When You Go Cheap'.",
                                answer_start=5, answer_end=6)

    instances = [instance1] # instance2
    dataset = Batch(instances)
    dataset.index_instances(model.vocab)
    # cuda_device = model._get_prediction_device()
    model_input = dataset.as_tensor_dict(verbose=True)
    outputs = model(**model_input)
    print(outputs["best_span_str"])

import json

def squad_subset():
    dataset_new = {"data": []}
    with open('/Users/daniel/ideaProjects/allennlp/allennlp/knn/data/squad-dev-v1.1.json') as f:
        dataset_json = json.load(f)
        dataset = dataset_json['data']
        dataset[0]['paragraphs'][0]["qas"] = dataset[0]['paragraphs'][0]["qas"][0:5]
        dataset[0]['paragraphs'][1]["qas"] = dataset[0]['paragraphs'][1]["qas"][0:5]
        dataset[0]['paragraphs'][2]["qas"] = dataset[0]['paragraphs'][2]["qas"][0:5]
        paragraphs_new = dataset[0]['paragraphs'][0:2]
    dataset_new["data"].append({"title": "Super_Bowl_50", "paragraphs": paragraphs_new})

    with open('/Users/daniel/ideaProjects/allennlp/allennlp/knn/data/squad-dev-v1.1-small-3.json', 'w') as outfile:
        json.dump(dataset_new, outfile)

# def openie_test():
#     from allennlp.predictors.predictor import Predictor
#     predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/openie-model.2018-08-20.tar.gz")
#     out = predictor.predict(
#         sentence="We first heard about this when the Google-Youtube acquisition news broke, and wrote briefly about it her"
#     )
#     print(out)


if __name__ == "__main__":
    train()
    # solve_sample_question()
    # openie_test()
    # squad_subset()

