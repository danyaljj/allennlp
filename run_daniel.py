#!/usr/bin/env python
import json
import logging
import os
import sys

import numpy
import numpy as np
import sklearn
import torch
from scipy import linalg
from torch import nn

import torch.nn.functional as F

from allennlp.common.util import ensure_list
from allennlp.data.dataset_readers import SquadReader
from allennlp.models.archival import load_archive, Archive
from allennlp.predictors import Predictor
from allennlp.data import DatasetReader
from allennlp.data.dataset import Batch
from sklearn import cluster
import time as time
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets.samples_generator import make_swiss_roll
from scipy.cluster.hierarchy import dendrogram, linkage

from evaluate11 import metric_max_over_ground_truths, f1_score, exact_match_score
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def solve(question, paragraph, model, dataset_reader, answers):
    instance = dataset_reader.text_to_instance(question, paragraph)
    instances = [instance]
    dataset = Batch(instances)
    dataset.index_instances(model.vocab)
    cuda_device = model._get_prediction_device()
    model_input = dataset.as_tensor_dict(cuda_device=cuda_device)
    outputs = model(**model_input)

    with open('out22-adv.txt', 'a') as ff:
        ff.write(question + "\n" + paragraph + "\n" + str(json.dumps(answers)) + "\n")
    # return outputs

def load_model():
    archive = load_archive("https://s3-us-west-2.amazonaws.com/allennlp/models/bidaf-model-2017.09.15-charpad.tar.gz")
    config = archive.config.duplicate()
    model = archive.model
    model.eval()
    dataset_reader_params = config["dataset_reader"]
    dataset_reader = DatasetReader.from_params(dataset_reader_params)

    return model, dataset_reader

def solve_sample_question():
    model, dataset_reader = load_model()

    question = "What kind of test succeeded on its first attempt?"
    paragraph = "One time I was writing a unit test, and it succeeded on the first attempt."

    a = solve(question, paragraph, model, dataset_reader, ["unit test"])
    print("")

def solve_squad_questions():
    model, dataset_reader = load_model()
    # dataset_file = "/Users/daniel/ideaProjects/linear-classifier/other/questionSets/squad-dev-v1.1.json"
    dataset_file = "/Users/daniel/ideaProjects/allennlp/sample1k-HCVerifySample.json"
    with open(dataset_file) as file:
        dataset_json = json.load(file)
        dataset = dataset_json['data']
        for article in dataset:
            for paragraph in article['paragraphs']:
                if len(paragraph['qas']) > 1:
                    continue
                else:
                    for qa in paragraph['qas']:
                        solve(qa['question'], paragraph['context'], model, dataset_reader, qa['answers'])

def filter_squad_questions():
    import json
    import copy
    dataset_file = "/Users/daniel/ideaProjects/allennlp/sample1k-HCVerifyAll.json"
    dataset_new = {"data": []}

    with open(dataset_file) as file:
        dataset_json = json.load(file)
        dataset = dataset_json['data']
        # dataset_copy = copy.deepcopy(dataset)
        for (aidx, article) in enumerate(dataset):
            # for (pid, paragraph) in enumerate(article['paragraphs']):
            # print("article: " + str(aidx) + " - paragraph: " + str(pid))
            # dataset_new["data"]
            paragraphs = dataset[aidx]['paragraphs']
            paragraphs_new = [x for x in paragraphs if len(x['qas']) <= 1]
            print("qas old: " + str(len(paragraphs)) + "  -  qas_new: " + str(len(paragraphs_new)))
            # article[aidx] = paragraphs_new
            dataset_new["data"].append({"paragraphs": paragraphs_new})
            # break
            # filtered_paragraphs = [x for x in article['paragraphs'] if len(article['paragraphs']) <= 1]
            # article['paragraphs'] = filtered_paragraphs

        with open('sample1k-HCVerifyAll-only-adversarial-instances.json', 'w') as outfile:
            json.dump(dataset_new, outfile)


def sample_clustering():
    k_clusters = 1
    from sklearn import cluster

    l = '[1, 1, 1, 1, 0, 0]'
    ll = json.loads(l) # [1, 1, 1, 1, 0, 0]
    mat = numpy.array([ll, ll, ll, ll])
    model = cluster.KMeans(n_clusters=k_clusters, n_init=200)
    model.fit(mat)
    print(list(model.labels_))

    cluster_to_idx_map = {}
    for i, cluster in enumerate(list(model.labels_)):
        if cluster not in cluster_to_idx_map:
            cluster_to_idx_map[cluster] = []
        cluster_to_idx_map[cluster].append(i)

    print(cluster_to_idx_map)

def cluster_predictions():
    activations_file = "/Users/daniel/ideaProjects/allennlp/out3.txt"
    questions_file = "/Users/daniel/ideaProjects/allennlp/out22.txt"

    features = []
    pred_ans = []
    with open(activations_file) as f:
        content = f.read().splitlines()
        for i, l in enumerate(content):
            # print(i)
            # print(l)
            if i % 2 == 0:
                # print(l)
                data = json.loads(l)
                # print(len(data))
                features.append(data)
            else:
                pred_ans.append(l)

    mat = numpy.array(features)


    k_clusters = 10
    model = cluster.KMeans(n_clusters=k_clusters)
    model.fit(mat)
    labels = model.labels_
    # fit = AgglomerativeClustering(n_clusters=6, linkage='ward').fit(mat)
    # labels = fit.labels_

    cluster_to_idx_map = {}
    for i, c in enumerate(list(labels)):
        if c not in cluster_to_idx_map:
            cluster_to_idx_map[c] = []
        cluster_to_idx_map[c].append(i)

    questions = []
    with open(questions_file) as f:
        content = f.read().splitlines()
        for i, l in enumerate(content):
            if i % 3 == 0:
                questions.append({})
                questions[int(i/3)]["q"] = l
            elif i % 3 == 1:
                questions[int(i/3)]["p"] = l
            elif i % 3 == 2:
                # print(l)
                ans = json.loads(l)
                questions[int(i/3)]["a"] = [c['text'] for c in ans]

    assert len(questions) == len(features), 'not the same number of activations and features ' + str(len(questions)) + ' vs ' + str(len(features))


    for c in cluster_to_idx_map.keys():
        instances = cluster_to_idx_map[c]
        print("-------------------")
        print("Questions in cluster: " + str(c) + " - cluster size : " + str(len(instances)))
        exact_match = 0
        f1 = 0
        score_size = 0
        for i in instances[0: 5]:
            print("Instance id: " + str(i))
            print(questions[i]["p"])
            print(questions[i]["q"])
            print("Gold: " + str(questions[i]["a"]))
            # print("Pred: " + str(pred_ans[i]))
            # print("-----")
            # a = solve(question, paragraph, model, dataset_reader)
            # ans = solve(questions[i]["q"], questions[i]["p"], model, dataset_reader)
            pred = pred_ans[i]
        #     # ground_truths = list(map(lambda x: x['text'], qa['answers']))
        #     # prediction = predictions[qa['id']]
            exact_match += metric_max_over_ground_truths(
                exact_match_score, pred, questions[i]["a"])
            f1 += metric_max_over_ground_truths(
                f1_score, pred, questions[i]["a"])
            score_size += 1
        exact_match = exact_match / score_size
        f1 = f1 / score_size
        print("exact_match: " + str(exact_match))
        print("f1: " + str(f1))

    # distance, weight = get_distances(mat, fit)
    # linkage_matrix = np.column_stack([fit.children_, distance, weight]).astype(float)
    # plt.figure(figsize=(20, 10))
    # dendrogram(linkage_matrix)
    # plt.show()

def get_distances(X,model,mode='l2'):
    distances = []
    weights = []
    children=model.children_
    dims = (X.shape[1],1)
    distCache = {}
    weightCache = {}
    for childs in children:
        c1 = X[childs[0]].reshape(dims)
        c2 = X[childs[1]].reshape(dims)
        c1Dist = 0
        c1W = 1
        c2Dist = 0
        c2W = 1
        if childs[0] in distCache.keys():
            c1Dist = distCache[childs[0]]
            c1W = weightCache[childs[0]]
        if childs[1] in distCache.keys():
            c2Dist = distCache[childs[1]]
            c2W = weightCache[childs[1]]
        d = np.linalg.norm(c1-c2)
        cc = ((c1W*c1)+(c2W*c2))/(c1W+c2W)

        X = np.vstack((X,cc.T))

        newChild_id = X.shape[0]-1

        # How to deal with a higher level cluster merge with lower distance:
        if mode=='l2':  # Increase the higher level cluster size suing an l2 norm
            added_dist = (c1Dist**2+c2Dist**2)**0.5
            dNew = (d**2 + added_dist**2)**0.5
        elif mode == 'max':  # If the previrous clusters had higher distance, use that one
            dNew = max(d,c1Dist,c2Dist)
        elif mode == 'actual':  # Plot the actual distance.
            dNew = d

        wNew = (c1W + c2W)
        distCache[newChild_id] = dNew
        weightCache[newChild_id] = wNew

        distances.append(dNew)
        weights.append( wNew)
    return distances, weights

def example_hierarchical_clustering():
    #############################################################################
    # Generate data (swiss roll dataset)
    n_samples = 1500
    noise = 0.05
    X, _ = make_swiss_roll(n_samples, noise)
    # Make it thinner
    X[:, 1] *= .5

    # #############################################################################
    # Compute clustering
    print("Compute unstructured hierarchical clustering...")
    st = time.time()
    ward = AgglomerativeClustering(n_clusters=6, linkage='ward').fit(X)
    elapsed_time = time.time() - st
    label = ward.labels_
    print("Elapsed time: %.2fs" % elapsed_time)
    print("Number of points: %i" % label.size)

    # #############################################################################
    # Plot result
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    ax.view_init(7, -80)
    for l in np.unique(label):
        ax.scatter(X[label == l, 0], X[label == l, 1], X[label == l, 2],
                   color=plt.cm.jet(np.float(l) / np.max(label + 1)),
                   s=20, edgecolor='k')
    plt.title('Without connectivity constraints (time %.2fs)' % elapsed_time)

    # #############################################################################
    # Define the structure A of the data. Here a 10 nearest neighbors
    from sklearn.neighbors import kneighbors_graph
    connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)

    # #############################################################################
    # Compute clustering
    print("Compute structured hierarchical clustering...")
    st = time.time()
    ward = AgglomerativeClustering(n_clusters=6, connectivity=connectivity,
                                   linkage='ward').fit(X)
    elapsed_time = time.time() - st
    label = ward.labels_
    print("Elapsed time: %.2fs" % elapsed_time)
    print("Number of points: %i" % label.size)

    # #############################################################################
    # Plot result
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    ax.view_init(7, -80)
    for l in np.unique(label):
        ax.scatter(X[label == l, 0], X[label == l, 1], X[label == l, 2],
                   color=plt.cm.jet(float(l) / np.max(label + 1)),
                   s=20, edgecolor='k')
    plt.title('With connectivity constraints (time %.2fs)' % elapsed_time)

    plt.show()

def find_eigen_values():
    activations_file = "/Users/daniel/ideaProjects/allennlp/out3.txt"
    questions_file = "/Users/daniel/ideaProjects/allennlp/out22.txt"

    features = []
    pred_ans = []
    with open(activations_file) as f:
        content = f.read().splitlines()
        for i, l in enumerate(content):
            # print(i)
            # print(l)
            if i % 2 == 0:
                # print(l)
                data = json.loads(l)
                # print(len(data))
                features.append(data)
            else:
                pred_ans.append(l)


    questions = []
    labels = []
    with open(questions_file) as f:
        content = f.read().splitlines()
        for i, l in enumerate(content):
            if i % 3 == 0:
                questions.append({})
                questions[int(i/3)]["q"] = l
            elif i % 3 == 1:
                questions[int(i/3)]["p"] = l
            elif i % 3 == 2:
                # print(l)
                ans = json.loads(l)
                questions[int(i/3)]["a"] = [c['text'] for c in ans]
                labels.append(ans[0]['text'])

    mat = numpy.array(features)
    mu = np.mean(mat, axis=0)
    mat_normalized = mat - mu

    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=5)
    # pca.fit(mat)
    # X_pca = pca.transform(mat)
    # print(X_pca.shape)

    # X_embedded = TSNE(n_components=2).fit_transform(mat)
    # print(X_embedded.shape)
    # fig, ax = plt.subplots()
    # plt.scatter(X_embedded[:, 0], X_embedded[:, 1])
    # plt.ylabel('Eigenvalues')
    # label = np.array(['label'])
    # plt.ylabel('Eigenvalues')
    # for i, txt in enumerate(labels):
    #     if  i % 40 == 0:
    #         ax.annotate(txt, (X_embedded[i, 0], X_embedded[i, 1]))
    # plt.show()

    # import csv
    # with open('qa_nn_tsne.csv', 'w', newline='') as csvfile:
    #     spamwriter = csv.writer(csvfile)
    #     for i, row in enumerate(X_embedded):
    #         row_tmp = [row[0], row[1], labels[i]]
    #         spamwriter.writerow(row_tmp)

    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=10)
    # pca.fit(mat)

    # def draw_vector(v0, v1, ax=None):
    #     ax = ax or plt.gca()
    #     arrowprops = dict(arrowstyle='->',
    #                       linewidth=2,
    #                       shrinkA=0, shrinkB=0)
    #     ax.annotate('', v1, v0, arrowprops=arrowprops)
    #
    # # plot data
    # plt.scatter(mat[:, 0], mat[:, 1], alpha=0.2)
    # for length, vector in zip(pca.explained_variance_, pca.components_):
    #     v = vector * 3 * np.sqrt(length)
    #     draw_vector(pca.mean_, pca.mean_ + v)
    # plt.axis('equal')

    # X_pca = pca.transform(mat)
    # print("original shape:   ", mat.shape)
    # print("transformed shape:", X_pca.shape)
    # X_new = pca.inverse_transform(X_pca)
    # plt.scatter(mat[:, 0], mat[:, 1], alpha=0.2)
    # plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
    # plt.axis('equal')
    # plt.show()

    U, s, Vh = linalg.svd(mat_normalized, full_matrices=False)

    # take the values corresponding to the i-th eigenvalue.
    # and take the top-k elements
    # and remember their indices

    import matplotlib.pyplot as plt


    # Projections on the i-th eigenvector
    i = 1
    a = np.transpose(U)[i]

    import csv
    with open('projection_onto_eg1-try2.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        for i, row in enumerate(a):
            qq = questions[i]["q"].split()[0:3]
            qPlusA = ' '.join(qq) + "? " + labels[i]
            row_tmp = [row, qPlusA]
            spamwriter.writerow(row_tmp)

    plt.hist(a, bins='auto')  # arguments are passed to np.histogram
    plt.title("Histogram of the projected values onto the third eigen-vector. ")
    plt.show()

    # sorted_indices_small_to_big = sorted(range(len(np.transpose(U)[0])), key=lambda k: np.transpose(U)[0][k])

    # UT = np.amax(abs(np.transpose(U)[0:10]), axis=0)
    # pca_scores = X_pca[:,0]
    # pca_scores = np.amax(abs(X_pca), axis=1)
    if False:
        sorted_indices_small_to_big = sorted(range(len(np.transpose(U)[999])), key=lambda k: np.transpose(U)[999][k])
        # sorted_indices_small_to_big = sorted(range(len(UT)), key=lambda k: UT[k])
        sorted_indices_small_to_big = sorted(range(len(pca_scores)), key=lambda k: pca_scores[k])
        top = sorted_indices_small_to_big[-10:]

        print("top: ")
        for i in top:
            print("--")
            print("Instance id: " + str(i))
            print(questions[i]["p"])
            print(questions[i]["q"])
            print("Gold: " + str(questions[i]["a"]))

        print("----------------")
        print("bottom: ")
        bottom = sorted_indices_small_to_big[:10]

        for i in bottom:
            print("--")
            print("Instance id: " + str(i))
            print(questions[i]["p"])
            print(questions[i]["q"])
            print("Gold: " + str(questions[i]["a"]))

        exact_match = []
        f1 = []

        for i in sorted_indices_small_to_big:
            print("--")
            print("Instance id: " + str(i))
            print(questions[i]["p"])
            print(questions[i]["q"])
            print("Gold: " + str(questions[i]["a"]))
            pred = pred_ans[i]
            exact_match.append(metric_max_over_ground_truths(
                exact_match_score, pred, questions[i]["a"]))
            f1.append(metric_max_over_ground_truths(
                f1_score, pred, questions[i]["a"]))

        N = 200
        cumsum, moving_aves = [0], []

        for i, x in enumerate(exact_match, 1):
            cumsum.append(cumsum[i - 1] + x)
            if i >= N:
                moving_ave = (cumsum[i] - cumsum[i - N]) / N
                # can do stuff with moving_ave here
                moving_aves.append(moving_ave)

        import matplotlib.pyplot as plt
        plt.plot(moving_aves)
        plt.ylabel('score (F1) ')
        plt.xlabel('Instances sorted based on the *maximum* of eigenvectors, corresponding to the top 10 eigenvalues (with PCA)')
        plt.ylim(ymax=1)  # adjust the max leaving min unchanged
        plt.ylim(ymin=0)  # adjust the min leaving max unchanged
        plt.show()

    # ymin, ymax = plt.ylim()  # return the current ylim
    # print(exact_match.tolist())

    # mat_matT = numpy.matmul(np.transpose(mat_normalized), mat_normalized)

    # print("asasd")

    # pca = sklearn.decomposition.PCA()
    # pca.fit(mat)

    # nComp = 2
    # Xhat = np.dot(pca.transform(mat)[:, :nComp], pca.components_[:nComp, :])
    # Xhat += mu
    #
    # print(Xhat[0,])

    # u, s, vh = np.linalg.svd(mat)
    # mat_matT = numpy.matmul(np.transpose(mat), mat)
    # eigenvalues = LA.eigvals(mat_matT)

    # import matplotlib.pyplot as plt
    # plt.plot(eigenvalues.tolist())
    # plt.ylabel('Eigenvalues')
    # plt.show()
    #
    # print(eigenvalues.tolist())

    # print("asasd")

def load_questions(activation_f, question_f, max_size = -1):
    activations_file = "/Users/daniel/ideaProjects/allennlp/" + activation_f
    questions_file = "/Users/daniel/ideaProjects/allennlp/" + question_f

    features = []
    pred_ans = []
    with open(activations_file) as f:
        content = f.read().splitlines()
        for i, l in enumerate(content):
            # print(i)
            # print(l[0:100])
            if i % 2 == 0:
                data = json.loads(l)
                # print(len(data))
                features.append(data)
            else:
                pred_ans.append(l)

            if(max_size > -1 and len(pred_ans) > max_size):
                break

    questions = []
    labels = []
    with open(questions_file) as f:
        content = f.read().splitlines()
        for i, l in enumerate(content):
            print(i)
            print(l[0:100])
            if i % 3 == 0:
                questions.append({})
                questions[int(i / 3)]["q"] = l
            elif i % 3 == 1:
                questions[int(i / 3)]["p"] = l
            elif i % 3 == 2:
                ans = json.loads(l)
                questions[int(i / 3)]["a"] = [c['text'] for c in ans]
                labels.append(ans[0]['text'])

            if (max_size > -1 and len(labels) > max_size):
                break
    mat = numpy.array(features)

    return (mat, labels, questions, pred_ans)

def project_adversarials_with_tsne():
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    (mat, labels, questions, pred_ans) = load_questions("out3.txt", "out22.txt", max_size=-1)
    (mat_ad, labels_ad, questions_ad, pred_ans_ad) = load_questions("out33-adv.txt", "out22-adv.txt", max_size=100)

    ones = numpy.ones(len(labels))
    zeros = numpy.zeros(len(labels_ad))

    mat_concat = np.concatenate((mat, mat_ad), axis=0)
    labels_concat = np.concatenate((labels, labels_ad))
    color_ids = np.concatenate((ones, zeros))
    color = ['red' if l == 0 else 'green' for l in color_ids]

    X_embedded = TSNE(n_components=2).fit_transform(mat_concat)
    print(X_embedded.shape)
    fig, ax = plt.subplots()
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], color=color, alpha=0.2)
    for i, txt in enumerate(labels_concat):
        print(i)
        if  i > len(labels): # or i % 25 == 0
            ax.annotate(txt, (X_embedded[i, 0], X_embedded[i, 1]), fontsize=7)
    plt.show()

    # import csv
    # with open('qa_nn_tsne_adv.csv', 'w', newline='') as csvfile:
    #     spamwriter = csv.writer(csvfile)
    #     for i, row in enumerate(X_embedded):
    #         row_tmp = [row[0], row[1], labels[i]]
    #         spamwriter.writerow(row_tmp)


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def printQuestionsForTypingTask():
    dataset_file = "/Users/daniel/ideaProjects/allennlp/squad-train-v1.1.json"
    questions = []
    with open(dataset_file) as file:
        dataset_json = json.load(file)
        dataset = dataset_json['data']
        for article in dataset[1:20]:
            for paragraph in article['paragraphs']:
                for qa in paragraph['qas']:
                    line = ''.join([qa['question'], " (Answer: ", qa['answers'][0]["text"], ")"])
                    questions.append(line)
            # break

    header  = ["question1", "question2", "question3", "question4", "question5", "question6", "question7", "question8", "question9", "question10"]

    import csv
    with open('mturk-input-2.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow(header)
        list = chunks(questions, 10)
        for row in list:
            if(len(row) == 10):
                spamwriter.writerow(row)

def processOutputOfMturk():
    # read json
    annotation_map = {} # map from question to their select labels

    def addQuestions(key, labels):
        if (key not in annotation_map):
            annotation_map[key] = []
        annotation_map[key].append(labels)

    import csv
    # with open('Batch_3333123_batch_results.csv') as csv_file:
    with open('Batch_3333447_batch_results.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                # print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
                line_count += 1
                questions = row[28:38]

                labels = row[38:-2]
                q1types = [x for x in labels if "q1-" in x]
                q2types = [x for x in labels if "q2-" in x]
                q3types = [x for x in labels if "q3-" in x]
                q4types = [x for x in labels if "q4-" in x]
                q5types = [x for x in labels if "q5-" in x]
                q6types = [x for x in labels if "q6-" in x]
                q7types = [x for x in labels if "q7-" in x]
                q8types = [x for x in labels if "q8-" in x]
                q9types = [x for x in labels if "q9-" in x]
                q10types = [x for x in labels if "q10-" in x]

                addQuestions(questions[0], q1types)
                addQuestions(questions[1], q2types)
                addQuestions(questions[2], q3types)
                addQuestions(questions[3], q4types)
                addQuestions(questions[4], q5types)
                addQuestions(questions[5], q6types)
                addQuestions(questions[6], q7types)
                addQuestions(questions[7], q8types)
                addQuestions(questions[8], q9types)
                addQuestions(questions[9], q10types)

    # extract the type of the question
    # print(annotation_map)

    from collections import Counter

    output = []

    for key in annotation_map.keys():
        print(key + str(annotation_map[key]))
        flatened_labels = [item for sublist in annotation_map[key] for item in sublist]
        list_of_labels = [x.split("-")[1] for x in flatened_labels]
        counts_of_labels = dict(Counter(list_of_labels))
        print(key)
        if len(key.split("(Answer:")) > 1:
            q = key.split("(Answer:")[0]
            a = key.split("(Answer:")[1][0:-1]
            output.append([q, a, counts_of_labels])

    with open('question_type_annotations.json', 'w') as outfile:
        json.dump(output, outfile)


if __name__ == "__main__":
    # solve_sample_question()
    # solve_squad_questions()
    # sample_clustering()
    # cluster_predictions()
    # example_hierarchical_clustering()
    # find_eigen_values()
    # project_adversarials_with_tsne()
    # filter_squad_questions()
    # printQuestionsForTypingTask()
    processOutputOfMturk()
