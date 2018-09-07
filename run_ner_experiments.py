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

model = "https://s3-us-west-2.amazonaws.com/allennlp/models/ner-model-2018.04.26.tar.gz"

def test_ner():
    model_url = "https://s3-us-west-2.amazonaws.com/allennlp/models/ner-model-2018.04.26.tar.gz"
    predictor = Predictor.from_path("")
    result = predictor.predict(sentence="this is a sentence")

def train_ner():
    pass

import os

def convert_ner_broad():
    file = "/Users/daniel/ideaProjects/allennlp/NER_datasets/broad_twitter/h.conll.txt"
    combined = convert_file_2_column(file)
    with open('/Users/daniel/ideaProjects/allennlp/NER_datasets/normalized-format/broad-h.txt', 'a') as ff:
        for line in combined:
            ff.write("\t".join(line) + "\n")

def convert_ner_muc():
    file = "/Users/daniel/ideaProjects/allennlp/NER_datasets/MUC7Columns/MUC7.NE.dryrun.sentences.columns.gold"
    combined = convert_file_9_column(file)
    with open('/Users/daniel/ideaProjects/allennlp/NER_datasets/normalized-format/muc-dev.txt', 'a') as ff:
        for line in combined:
            ff.write("\t".join(line) + "\n")

def convert_ner_ontonotes():
    folder = "/Users/daniel/ideaProjects/allennlp/NER_datasets/ontonotes/ColumnFormat/Test/"
    arr = os.listdir(folder)
    combined = []
    for cont in arr:
        combined.extend(convert_file_9_column(folder + cont))
    with open('/Users/daniel/ideaProjects/allennlp/NER_datasets/normalized-format/onotnotes-dev.txt', 'a') as ff:
        for line in combined:
            ff.write("\t".join(line) + "\n")

valid_tags = ["B-PER", "I-PER", "B-ORG", "I-ORG", "I-LOC", "B-LOC"]
ontonotes_tag_conversion = { "B-PERSON": "I-PER", "I-PERSON": "I-PER" }

def get_label(label):
    if("-" in label):
        return label.split("-")[1]
    return label

def have_same_label(label1, label2):
    return get_label(label1) == get_label(label2)

def convert_bio2_to_bio1(lines):
    for i, line in enumerate(lines):
        if i > 0:
            # print(lines[i], len(lines[i]) <= 1)
            # print(lines[i])
            if len(lines[i]) > 1 and ("B-" in lines[i][3]) and (len(lines[i-1]) <= 1 or lines[i-1][3] == "O" or not have_same_label(lines[i-1][3], lines[i][3])):
                lines[i][3] = "I-" + lines[i][3].split("-")[1] # change the label to be "inside"
    return lines

def convert_file_9_column(file):
    filtered_lines = []
    with open(file) as f:
        content = f.readlines()
        for line in content:
            split = line.split("\t")
            if len(split) > 1:
                # print(split)
                if split[0] in ontonotes_tag_conversion:
                    new_label = ontonotes_tag_conversion[split[0]]
                else:
                    new_label = split[0]

                if new_label not in valid_tags:
                    new_label = "O"
                # print(new_label)

                new_line = [split[5], ".", ".", new_label]

                filtered_lines.append(new_line)
            else:
                filtered_lines.append(line)

    converted_lines = convert_bio2_to_bio1(filtered_lines)
    return converted_lines

def convert_file_2_column(file):
    filtered_lines = []
    with open(file) as f:
        content = f.readlines()
        for line in content:
            split = line.replace("\n", "").split("\t")
            # print(split)
            if len(split) > 1:
                # print(split)
                # if split[1] in ontonotes_tag_conversion:
                #     new_label = ontonotes_tag_conversion[split[1]]
                # else:
                new_label = split[1]
                # print(new_label)
                if new_label not in valid_tags:
                    new_label = "O"
                if new_label == "B-PER":
                    new_label = "I-PER"
                # print(new_label)

                new_line = [split[0], ".", ".", new_label]

                filtered_lines.append(new_line)
            # else:
            #     filtered_lines.append(line)

    converted_lines = convert_bio2_to_bio1(filtered_lines)
    return converted_lines

# def plot_qualities():



if __name__ == "__main__":
    # test_ner()
    # convert_ner_ontonotes()
    # convert_ner_muc()
    convert_ner_broad()


