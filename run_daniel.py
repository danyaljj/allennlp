import json
import logging
import os
import sys

import numpy
import numpy as np
import re
from sklearn.metrics import confusion_matrix
from scipy import linalg
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
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

from evaluate11 import metric_max_over_ground_truths, f1_score, exact_match_score
import numpy as np
from numpy import linalg as LA
import pickle
from allennlp.modules.elmo import Elmo, batch_to_ids

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def solve(question, paragraph, model, dataset_reader, answers):
    print(question)
    print(paragraph)
    print(answers)
    instance = dataset_reader.text_to_instance(question, paragraph)
    instances = [instance]
    dataset = Batch(instances)
    dataset.index_instances(model.vocab)
    cuda_device = model._get_prediction_device()
    model_input = dataset.as_tensor_dict(cuda_device=cuda_device)
    outputs = model(**model_input)

    with open('ipython/mctest/out22-ner-test.txt', 'a') as ff:
        ff.write(question.replace('\n', ' ') + "\n" + paragraph.replace('\n', ' ') + "\n" + str(json.dumps(answers)) + "\n")

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

def solve_sample_question():
    model, dataset_reader = load_model()

    question = "What kind of test succeeded on its first attempt?"
    paragraph = "One time I was writing a unit test, and it succeeded on the first attempt."

    a = solve(question, paragraph, model, dataset_reader, ["unit test"])
    print("")

def solve_squad_questions():
    model, dataset_reader = load_model()
    # dataset_file = "/Users/daniel/ideaProjects/linear-classifier/other/questionSets/squad-dev-v1.1.json"
    # dataset_file = "/Users/daniel/ideaProjects/allennlp/ontonotes_questions_ner.json"
    # dataset_file = "/Users/daniel/ideaProjects/allennlp/ontonotes_questions_ner_test_full.json"
    # "/Users/daniel/ideaProjects/allennlp/sample1k-HCVerifySample.json"
    dataset_file = "/Users/daniel/ideaProjects/allennlp/QA_datasets/mctest.json"
    # dataset_file = "/Users/daniel/ideaProjects/allennlp/QA_datasets/mutlirc_questions.json"
    # dataset_file = "/Users/daniel/ideaProjects/allennlp/babi-test.json" # "/Users/daniel/ideaProjects/allennlp/sample1k-HCVerifySample.json"
    # dataset_file = "/Users/daniel/ideaProjects/linear-classifier/other/questionSets/cachedQuestions/process-bank-train.json"
    # dataset_file = "/Users/daniel/ideaProjects/linear-classifier/other/questionSets/cachedQuestions/remedia-questions.json"
    # dataset_file = "/Users/daniel/ideaProjects/linear-classifier/other/questionSets/cachedQuestions/Public-Feb2016-Elementary-NDMC-Train.json"
    with open(dataset_file) as file:
        dataset_json = json.load(file)
        dataset = dataset_json['data']
        for article in dataset:
            for i, paragraph in enumerate(article['paragraphs']):
                print("Progress: " + str(100.0 * i / len(article['paragraphs'])))
                if False and len(paragraph['qas']) > 1:
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

            if (max_size > -1 and len(pred_ans) > max_size):
                break

            if i % 2 == 0:
                # print(l)
                data = json.loads(l)
                features.append(data)
            else:
                pred_ans.append(l)



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

    (mat, labels, questions, pred_ans) = load_questions("out3.txt", "out22.txt", max_size=0)
    (mat_ad, labels_ad, questions_ad, pred_ans_ad) = load_questions("out33-adv.txt", "out22-adv.txt", max_size=200)

    # ones = numpy.ones(len(labels))
    # zeros = numpy.zeros(len(labels_ad))

    ones = numpy.ones(len(pred_ans))
    zeros = numpy.zeros(len(pred_ans_ad))

    mat_concat = np.concatenate((mat, mat_ad), axis=0)
    labels_concat = np.concatenate((labels, labels_ad))
    color_ids = np.concatenate((ones, zeros))
    color = ['red' if l == 0 else 'green' for l in color_ids]

    X_embedded = TSNE(n_components=2,init="pca").fit_transform(mat_concat)
    # print(X_embedded.shape)
    fig, ax = plt.subplots()
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], color=color, alpha=0.6, s=1.1)
    for i, txt in enumerate(labels_concat):
        print(i)
        if  i > len(labels): # i % 10 < 1:
            ax.annotate(txt, (X_embedded[i, 0] * 0.98, X_embedded[i, 1]), fontsize=6.5)
    plt.show()

    # import csv
    # with open('qa_nn_tsne_adv.csv', 'w', newline='') as csvfile:
    #     spamwriter = csv.writer(csvfile)
    #     for i, row in enumerate(X_embedded):
    #         row_tmp = [row[0], row[1], labels[i]]
    #         spamwriter.writerow(row_tmp)

def project_remedia_with_tsne():
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    (mat, labels, questions, pred_ans) = load_questions("out3.txt", "out22.txt", max_size=1200)
    (mat_ad, labels_ad, questions_ad, pred_ans_ad) = load_questions("out33-rem.txt", "out22-rem.txt", max_size=100)

    ones = numpy.ones(len(pred_ans))
    zeros = numpy.zeros(len(pred_ans_ad))


    question_initials = [" ".join(x["q"].split(" ")[:2]) for x in questions]
    question_initials_ad = [" ".join(x["q"].split(" ")[:2]) for x in questions_ad]
    for i, x in enumerate(labels):
        labels[i] = question_initials[i] + "//" + labels[i]
    for i, x in enumerate(labels_ad):
        labels_ad[i] = question_initials_ad[i] + "//" + labels_ad[i]
    mat_concat = np.concatenate((mat, mat_ad), axis=0)
    labels_concat = np.concatenate((labels, labels_ad))
    color_ids = np.concatenate((ones, zeros))
    color = ['red' if l == 0 else 'green' for l in color_ids]

    X_embedded = TSNE(n_components=2,init="pca").fit_transform(mat_concat)
    # print(X_embedded.shape)
    fig, ax = plt.subplots()
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], color=color, alpha=0.6, s=1.1)
    for i, txt in enumerate(labels_concat):
        print(i)
        if  i > len(labels): # i % 10 < 1:
            ax.annotate(txt, (X_embedded[i, 0] * 0.98, X_embedded[i, 1]), fontsize=6.5)
    plt.show()

import matplotlib
from matplotlib.cm import cool

def get_n_colors(n):
    return[ cool(float(i)/n) for i in range(n) ]


def project_babi_with_tsne():
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    (mat, labels, questions, pred_ans) = load_questions("out3.txt", "out22.txt", max_size=0)
    (mat_ad, labels_ad, questions_ad, pred_ans_ad) = load_questions("out33-adv.txt", "out22-adv.txt", max_size=-1)

    # read the questions and remember question ids:
    question_paragraph_id_map = {}
    ff = "/Users/daniel/ideaProjects/allennlp/QA_datasets/babi-test.json"
    with open(ff) as file:
        dataset_json = json.load(file)
        dataset = dataset_json['data']
        for article in dataset:
            for paragraph in article['paragraphs']:
                for qa in paragraph['qas']:
                    key = qa['question'] + paragraph['context']
                    id = qa['id']
                    question_paragraph_id_map[key] = id

    bibi_reasoning_types = [question_paragraph_id_map[x["q"]+x["p"]] for x in questions_ad]

    # find the quality per reasoning type
    for c in set(bibi_reasoning_types):
        predictions_a = np.array([pred_ans_ad[i] for i, r in enumerate(bibi_reasoning_types) if r == c])
        labels_a = np.array([labels_ad[i] for i, r in enumerate(bibi_reasoning_types) if r == c])
        similarity = [f1_score(x,y) for x,y in zip(predictions_a, labels_a)]
        assert len(predictions_a) == len(labels_a)
        print(str(c) + " -> " + str(100.0 * sum(similarity) / len(predictions_a)))

    similarity = [f1_score(x,y) for x,y in zip(pred_ans, labels)]
    # print("Squad -> ", str(100.0 * sum(similarity) / len(pred_ans)))


    mat_concat = np.concatenate((mat, mat_ad), axis=0)
    reasoning_types = ["squad"]*len(pred_ans) + bibi_reasoning_types
    unique_reasoning_types = list(set(bibi_reasoning_types))
    # unique_reasoning_types_indices = list(range(len(unique_reasoning_types)))
    bibi_reasoning_type_indices = [unique_reasoning_types.index(x) for x in bibi_reasoning_types]
    labels_concat = np.concatenate((labels, labels_ad))
    pred_concat = np.concatenate((pred_ans, pred_ans_ad))

    # X_embedded = TSNE(n_components=2,init="pca").fit_transform(mat_concat)
    # fig, ax = plt.subplots()
    # for iter, c in enumerate(unique_reasoning_types):
    #     X_selected = np.asarray([X_embedded[i, :] for i, r in enumerate(reasoning_types) if r == c and f1_score(pred_concat[i], labels_concat[i]) > 0.6])
    #     plt.scatter(X_selected[:, 0], X_selected[:, 1], alpha=0.7, s=2, label=c) # color=color_map[iter],
    #
    # ax.legend()
    # plt.show()

    import seaborn as sns

    db = KMeans(n_clusters=len(unique_reasoning_types), random_state=2).fit(mat_ad)
    # db = DBSCAN(eps=0.3, min_samples=10).fit(mat_ad)
    # core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    # core_samples_mask[db.core_sample_indices_] = True
    labelsss = db.labels_

    matt1 = confusion_matrix(bibi_reasoning_type_indices, labelsss)

    # for i in len(matt):
    #     pass

    # sort the labels based on their
    # sorted_labels = []
    # for

    matt = confusion_matrix(bibi_reasoning_type_indices, labelsss)
    max_values = np.max(matt1, axis=1)

    # matt = diagonalize(matt)
    plt.figure(figsize=(5.5, 4))
    sss = sns.heatmap(matt.T, square=True,
                annot=True,
                fmt='d', cbar=False,
                xticklabels=True,
                yticklabels=True
                )
    sss.set_xticklabels(unique_reasoning_types, rotation=45)
    sss.set_yticklabels(unique_reasoning_types, rotation=0)
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.show()
    print(len(unique_reasoning_types))

    ars = metrics.adjusted_rand_score(labelsss, bibi_reasoning_type_indices)
    print("ars: " + str(ars))


def diagonalize(A):
    # A = np.matrix([[-1, -1, 5], [1, 3, -1], [3, 1, -1]])
    A = A.astype(int)
    maxIndices = []
    for i in range(len(A)):
        maxIndices.append(np.argmax(A[:,i]))
    B = A.copy()
    print(A.shape)
    print(B.shape)

    indexmaps = list(enumerate(maxIndices))
    indexmaps.sort(key=lambda x: x[1])
    sorted_indices = [xx[0] for xx in indexmaps]

    for oldId, newId in enumerate(sorted_indices):
        # B[:, i] = A[:, i]
        for j in range(len(A)):
            B[j,newId] = A[j,oldId]

    return B


def project_ner_with_tsne():
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    (mat, labels, questions, pred_ans) = load_questions("out3.txt", "out22.txt", max_size=0)
    (mat_ad, labels_ad, questions_ad, pred_ans_ad) = load_questions("out33-ner.txt", "out22-ner.txt", max_size=3000)

    # read the questions and remember question ids:
    question_paragraph_id_map = {}
    ff = "/Users/daniel/ideaProjects/allennlp/ontonotes_questions_ner.json"
    with open(ff) as file:
        dataset_json = json.load(file)
        dataset = dataset_json['data']
        for article in dataset:
            for paragraph in article['paragraphs']:
                for qa in paragraph['qas']:
                    key = qa['question'] + paragraph['context']
                    id = qa['id']
                    question_paragraph_id_map[key] = id

    ner_label_per_question = [question_paragraph_id_map[x["q"]+x["p"]] for x in questions_ad]

    target_labels = [
        "NORP",
        "GPE",
        "LOC",
        "EVENT",
        "TIME",
        "CARDINAL",
        "WORK_OF_ART",
        "ORDINAL",
        "MONEY",
        "ORG",
        "PERSON",
        "DATE"
    ]

    # find the quality per reasoning type
    # for c in set(ner_label_per_question):
    #     predictions_a = np.array([pred_ans_ad[i] for i, r in enumerate(ner_label_per_question) if r == c])
    #     labels_a = np.array([labels_ad[i] for i, r in enumerate(ner_label_per_question) if r == c])
    #     similarity = [f1_score(x,y) for x,y in zip(predictions_a, labels_a)]
    #     assert len(predictions_a) == len(labels_a)
    #     print(str(c) + " -> " + str(100.0 * sum(similarity) / len(predictions_a)))
    #
    # similarity = [f1_score(x,y) for x,y in zip(pred_ans, labels)]
    # print("Squad -> " + str(100.0 * sum(similarity) / len(pred_ans)))

    mat_concat = np.concatenate((mat, mat_ad), axis=0)
    reasoning_types = ["squad"]*len(pred_ans) + ner_label_per_question
    unique_ner_label = set(reasoning_types)
    labels_concat = np.concatenate((labels, labels_ad))
    pred_concat = np.concatenate((pred_ans, pred_ans_ad))

    X_embedded = TSNE(n_components=2,init="pca").fit_transform(mat_concat)
    fig, ax = plt.subplots()
    for iter, c in enumerate(target_labels):
        X_selected = np.asarray([X_embedded[i, :] for i, r in enumerate(reasoning_types) if r == c]) # and f1_score(pred_concat[i], labels_concat[i]) > 0.6
        plt.scatter(X_selected[:, 0], X_selected[:, 1], alpha=1.0, s=4.5, label=c)

    ax.legend()
    plt.show()


def dist(x, y):
    return np.linalg.norm(x - y)


def ner_few_shot():
    # load instances
    (mat_ad, labels_ad, questions_ad, pred_ans_ad) = load_questions("out33-ner-test.txt", "out22-ner-test.txt", max_size=1000000)

    print("instances loaded: " + str(len(questions_ad)))

    # read the questions and remember question ids:
    question_paragraph_id_map = {}
    ff = "/Users/daniel/ideaProjects/allennlp/ontonotes_questions_ner_test_full.json"
    with open(ff) as file:
        dataset_json = json.load(file)
        dataset = dataset_json['data']
        for article in dataset:
            for paragraph in article['paragraphs']:
                for qa in paragraph['qas']:
                    key = qa['question'] + paragraph['context']
                    id = qa['id']
                    question_paragraph_id_map[key] = id
    ner_label_per_question = [question_paragraph_id_map[x["q"] + x["p"]] for x in questions_ad]

    # select representative instances
    seen_size = 5000
    seen_mat = mat_ad[0:seen_size]
    seen_labels = ner_label_per_question[0:seen_size]

    unseen_mat = mat_ad[seen_size:]
    unseen_labels = ner_label_per_question[seen_size:]

    target_labels = [
        "NORP",
        "GPE",
        "LOC",
        "EVENT",
        "TIME",
        "CARDINAL",
        "WORK_OF_ART",
        "ORDINAL",
        "MONEY",
        "ORG",
        "PERSON",
        "DATE",
        "LANGUAGE",
        "QUANTITY",
        "PERCENT",
        "FAC",
        "LAW",
        "PRODUCT"
    ]

    # find averaged centers
    centers = {}
    for l in target_labels:
        selected = np.asarray([seen_mat[i, :] for i, r in enumerate(seen_labels) if r == l])[0:5]
        print("Number of representative instances for label " + l + " -> " + str(len(selected)))
        centers[l] = numpy.mean(selected, axis=0)
        # print(len(seen_mat[0, :]))
        # print(len(centers[l]))
        assert len(centers[l]) == len(seen_mat[0, :])

    tp_per_label = { l:0 for l in target_labels }
    fp_per_label = { l:0 for l in target_labels }
    fn_per_label = { l:0 for l in target_labels }

    # do classification
    print("Size of the test collection: " + str(len(unseen_mat)))
    for i,m in enumerate(unseen_mat):
        gold_label = unseen_labels[i]
        if gold_label in target_labels:
            predicted_label = -1
            min_dist = numpy.math.inf
            for l in centers.keys():
                new_dist = dist(m, centers[l])
                if new_dist < min_dist:
                    min_dist = new_dist
                    predicted_label = l
            assert predicted_label != -1
            if gold_label == predicted_label:
                tp_per_label[gold_label] = tp_per_label[gold_label] + 1
            else:
                fp_per_label[predicted_label] = fp_per_label[predicted_label] + 1
                fn_per_label[gold_label] = fn_per_label[gold_label] + 1

    print("TP: " + str(tp_per_label))
    print("FP: " + str(fp_per_label))
    print("FN: " + str(fn_per_label))

    for l in target_labels:
        tp = tp_per_label[l]
        fp = fp_per_label[l]
        fn = fn_per_label[l]
        # print("l: " + str(l))
        # print("tp: " + str(tp))
        # print("fp: " + str(fp))
        # print("fn: " + str(fn))
        if tp + fp == 0:
            p = 0.0
        else:
            p = 100.0 * tp / (tp + fp)

        if tp + fn == 0:
            r = 0.0
        else:
            r = 100.0 * tp / (tp + fn)
        if p + r == 0.0:
            f1 = 0.0
        else:
            f1 = 2 * p * r / (p + r)
        print("Label: " + str(l) + "\t" + str(p) + "\t" + str(r) + "\t" + str(f1))

def typing_classifier():

    elmoCache = ELMoCache()
    elmoCache.load_from_disk()

    # load instances
    max_size = 10000
    (bidaf_vectors, bidaf_pred, questions, pred_ans_ad) = load_questions("out33-ner-test.txt", "out22-ner-test.txt", max_size=max_size)

    target_labels = [
        "NORP",
        "GPE",
        "LOC",
        "EVENT",
        "TIME",
        "CARDINAL",
        "WORK_OF_ART",
        "ORDINAL",
        "MONEY",
        "ORG",
        "PERSON",
        "DATE",
        "LANGUAGE",
        "QUANTITY",
        "PERCENT",
        "FAC",
        "LAW",
        "PRODUCT"
    ]

    # read the questions and remember question ids:
    question_paragraph_id_map = {}
    ff = "/Users/daniel/ideaProjects/allennlp/ontonotes_questions_ner_test_full.json"
    with open(ff) as file:
        dataset_json = json.load(file)
        dataset = dataset_json['data']
        for article in dataset:
            for paragraph in article['paragraphs']:
                for qa in paragraph['qas']:
                    key = qa['question'] + paragraph['context']
                    id = qa['id']
                    question_paragraph_id_map[key] = id
    ner_label_per_question = [question_paragraph_id_map[x["q"] + x["p"]] for x in questions]

    # select representative instances
    bidaf_vectors = [bidaf_vectors[i,:] for i, l in enumerate(ner_label_per_question) if l in target_labels]
    questions = [questions[i] for i, l in enumerate(ner_label_per_question) if l in target_labels]
    ner_label_per_question = [ner_label_per_question[i] for i, l in enumerate(ner_label_per_question) if l in target_labels]
    # seen_size = int(0.5 * len(questions))
    seen_size = int(len(questions) * 0.3)

    # print("seen size before filtering: " + str(seen_size))
    # max_per_label = 1000
    # count_per_label = {l: 0 for l in target_labels}
    # selected_indices = []
    # for i,l in enumerate(ner_label_per_question):
    #     if count_per_label[l] < max_per_label:
    #         count_per_label[l] = count_per_label[l] + 1
    #         selected_indices.append(i)

    # bidaf_vectors = [bidaf_vectors[i] for i, l in enumerate(bidaf_vectors) if i in selected_indices or i > max(selected_indices)]
    # questions = [questions[i] for i, l in enumerate(questions) if i in selected_indices or i > max(selected_indices)]
    # ner_label_per_question = [ner_label_per_question[i] for i, l in enumerate(ner_label_per_question) if i in selected_indices or i > max(selected_indices)]

    # find averaged centers
    centers = {}
    for l in target_labels:
        selected = np.asarray([bidaf_vectors[i] for i in range(0,len(ner_label_per_question)) if ner_label_per_question[i] == l])
        print("Number of representative instances for label " + l + " -> " + str(len(selected)))
        centers[l] = numpy.mean(selected, axis=0)
        # print(len(seen_mat[0, :]))
        # print(len(centers[l]))
        assert len(centers[l]) == len(bidaf_vectors[0])

    def dist_to_centers(vec):
        dist_vec = []
        for l in centers.keys():
            dist_vec.append(dist(vec, centers[l]))
        # add the index of the minimum and 2nd minimum label
        sorted_indices = sorted(range(len(dist_vec)), key=lambda k: dist_vec[k])  # indices of the distance, sorted, from smallest to the biggest
        sorting_encoding = []
        for idx in sorted_indices:
            zero_vector = numpy.zeros(len(dist_vec))
            zero_vector[idx] = 1
            sorting_encoding.extend(zero_vector)
        return dist_vec + sorting_encoding

    elmo_vectors = []
    elmo_bidaf_vectors = []
    for i, x in enumerate(questions):
        paragraph_vec = elmoCache.get_elmo([x["p"].split(" ")])
        span_vec = elmoCache.get_elmo([bidaf_pred[i].split(" ")])
        elmo_vec = paragraph_vec + span_vec
        elmo_vectors.append(elmo_vec)
        # print(len(list(bidaf_vec[i, :])))
        bidaf_vec = list(bidaf_vectors[i]) + dist_to_centers(bidaf_vectors[i])
        bidaf_vectors[i] = bidaf_vec
        elmo_bidaf_vectors.append(elmo_vec + bidaf_vec)
        if i % 100 == 0:
            print(" - Processed " + str(100.0 * i/len(questions)))

    elmoCache.save_to_disk()

    # selected_size = len(selected_indices)
    seen_labels = ner_label_per_question[0:seen_size]
    unseen_labels = ner_label_per_question[seen_size:]
    print("seen size before filtering: " + str(seen_size))

    seen_elmo_vectors = elmo_vectors[0:seen_size]
    unseen_elmo_vectors = elmo_vectors[seen_size:]

    seen_bidaf_vectors = bidaf_vectors[0:seen_size]
    unseen_bidaf_vectors = bidaf_vectors[seen_size:]

    seen_elmo_bidaf_vectors = elmo_bidaf_vectors[0:seen_size]
    unseen_elmo_bidaf_vectors = elmo_bidaf_vectors[seen_size:]

    # elmo_mlp = MLPClassifier()
    elmo_mlp = MLPClassifier() #LogisticRegression(multi_class="multinomial") #
    elmo_mlp.fit(seen_elmo_vectors, seen_labels)
    elmo_predictions = elmo_mlp.predict(unseen_elmo_vectors)

    bidaf_mlp = MLPClassifier() #LogisticRegression(multi_class="multinomial") # MLPClassifier()
    bidaf_mlp.fit(seen_bidaf_vectors, seen_labels)
    bidaf_predictions = bidaf_mlp.predict(unseen_bidaf_vectors)

    elmo_bidaf_mlp = MLPClassifier() # LogisticRegression(multi_class="multinomial") # MLPClassifier()
    elmo_bidaf_mlp.fit(seen_elmo_bidaf_vectors, seen_labels)
    elmo_bidaf_predictions = elmo_bidaf_mlp.predict(unseen_elmo_bidaf_vectors)

    def evaluate(pred, gold):
        tp_per_label = { l:0 for l in target_labels }
        fp_per_label = { l:0 for l in target_labels }
        fn_per_label = { l:0 for l in target_labels }

        # do classification
        for i,m in enumerate(pred):
            gold_label = gold[i]
            predicted_label = pred[i]
            if gold_label == predicted_label:
                tp_per_label[gold_label] = tp_per_label[gold_label] + 1
            else:
                fp_per_label[predicted_label] = fp_per_label[predicted_label] + 1
                fn_per_label[gold_label] = fn_per_label[gold_label] + 1

        print("TP: " + str(tp_per_label))
        print("FP: " + str(fp_per_label))
        print("FN: " + str(fn_per_label))

        for l in target_labels:
            tp = tp_per_label[l]
            fp = fp_per_label[l]
            fn = fn_per_label[l]
            if tp + fp == 0:
                p = 0.0
            else:
                p = 100.0 * tp / (tp + fp)

            if tp + fn == 0:
                r = 0.0
            else:
                r = 100.0 * tp / (tp + fn)

            if p + r == 0.0:
                f1 = 0.0
            else:
                f1 = 2 * p * r / (p + r)
            print("Label: " + str(l) + "\t" + str(p) + "\t" + str(r) + "\t" + str(f1))

    print(len(seen_elmo_vectors))
    print(len(unseen_elmo_vectors))
    print(len(seen_labels))
    print(len(unseen_elmo_vectors))

    print("ELMO predictions: ")
    evaluate(elmo_predictions, unseen_labels)

    print("Bidaf predictions: ")
    evaluate(bidaf_predictions, unseen_labels)

    print("ELMO + BiDAF predictions")
    evaluate(elmo_bidaf_predictions, unseen_labels)


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
    lines = []
    with open('Batch_3333123_batch_results.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            lines.append(row)

    with open('Batch_3333447_batch_results.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            lines.append(row)

    line_count = 0
    for row in lines :
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
        if len(key.split("(Answer:")) > 1 and len(list_of_labels) > 0:
            q = key.split("(Answer:")[0]
            a = key.split("(Answer:")[1][0:-1]
            output.append([q, a, counts_of_labels])

    with open('question_type_annotations.json', 'w') as outfile:
        json.dump(output, outfile)

def filter_questions_answer_types():
    dataset_file = "/Users/daniel/ideaProjects/allennlp/squad-train-v1.1.json"

    from ccg_nlpy import remote_pipeline
    pipeline = remote_pipeline.RemotePipeline(server_api="http://macniece.seas.upenn.edu:4002")

    dataset_new = {"data": []}

    import copy

    with open(dataset_file) as file:
        dataset_json = json.load(file)
        dataset = dataset_json['data']
        for article in dataset:
            paragraphs_new = []
            for p in article['paragraphs']:
                p_new = copy.deepcopy(p)
                p_new["qas"] = []

                for qa in p['qas']:
                    a = qa['answers'][0]["text"]
                    try:
                        d = pipeline.doc(a)
                        # print(a + " - " + str(d.get_ner_ontonotes) + " - " + str(d.get_ner_conll))
                        # d.get_ner_conll

                        conll_vu = d.get_ner_conll
                        ontonotes_vu = d.get_ner_ontonotes

                        if conll_vu.cons_list is None:
                            conll_labels = []
                        else:
                            # print(conll_vu)
                            conll_labels = set([x['label'] for x in conll_vu.cons_list])

                        if ontonotes_vu.cons_list is None:
                            onto_labels = []
                        else:
                            onto_labels = set([x['label'] for x in ontonotes_vu.cons_list])


                        if ("PER" in conll_labels and len(conll_labels) == 1) or ("PERSON" in onto_labels and len(onto_labels) == 1):
                            print("valid 'person' question: "  + str(qa))
                            p_new["qas"].append(qa)

                        # if ("ORG" in conll_labels and len(conll_labels) == 1) or ("ORG" in onto_labels and len(onto_labels) == 1):
                        #     print("valid 'organization' question: "  + str(qa))
                        #     p_new["qas"].append(qa)

                        # if "DATE" in onto_labels and len(onto_labels) == 1:
                        #     print("valid 'date' question: "  + str(qa))
                        #     p_new["qas"].append(qa)
                    except Exception:
                        print("Exception . . . ")
                        print(a)

                if len(p_new["qas"]) > 0:
                    paragraphs_new.append(p_new)
            dataset_new["data"].append({"paragraphs": paragraphs_new})
            # break

        with open('squad_train_person_only.json', 'w') as outfile:
            json.dump(dataset_new, outfile)

import os
import os.path

def find_str(s, char):
    index = 0

    if char in s:
        c = char[0]
        for ch in s:
            if ch == c:
                if s[index:index+len(char)] == char:
                    return index

            index += 1

    return -1


def get_files(target_dir):
    item_list = os.listdir(target_dir)

    file_list = list()
    for item in item_list:
        item_dir = os.path.join(target_dir,item)
        if os.path.isdir(item_dir):
            file_list += get_files(item_dir)
        else:
            file_list.append(item_dir)
    return file_list

def load_babi_questions():
    folder = "/Users/daniel/ideaProjects/linear-classifier/other/questionSets/babi_tasks_1-20_v1-2/en-10k/"

    files = get_files(folder)

    train_files = [x for x in files if "train.txt" in x]
    test_files = [x for x in files if "test.txt" in x]

    def read(file_list):
        paragraphs = []
        for ff in file_list:
            with open(ff) as f:
                file_name = ff.split("_")[-2]
                content = f.read().splitlines()
                sentences = ""
                qas = []
                for i, line in enumerate(content):
                    split = re.compile("^\d{1,3}|\t").split(line)
                    # print(line)
                    # print(content[i+1][0])
                    # print(split)
                    # split =  line.split("\t")
                    # paragraph sentence:
                    if len(split) == 2:
                        sentences = (sentences + " " + split[1].strip()).strip()
                    else:
                        # question and answer
                        ans_text = split[2].strip()
                        idx = find_str(sentences, ans_text)
                        if idx >= 0:
                            ans = [{"answer_start": idx, "text": ans_text}]
                            question = {"answers": ans, "question": split[1].strip(), "id": file_name}
                            qas.append(question)
                        # end of paragraph; add the questions to the list
                    if i + 1 < len(content) and content[i+1][0] == "1":
                        if len(qas) > 0:
                            paragraphs.append({"context": sentences.strip(), "qas": qas})
                        sentences = ""
                        qas = []

        return {"data": [{"paragraphs": paragraphs}]}

    train_paragraphs = read(train_files)
    test_paragraphs = read(test_files)
    # return (train_paragraphs, test_paragraphs)
    with open('babi-train.json', 'w', newline='') as f:
        f.write(json.dumps(train_paragraphs))
    with open('QA_datasets/babi-test.json', 'w', newline='') as f:
        f.write(json.dumps(test_paragraphs))

class ELMoCache():
    cache = {}

    def __init__(self, cache_file = "elmo.cache"):
        self.cache_file = cache_file
        self.elmo = load_elmo()

    def load_from_disk(self):
        with open(self.cache_file, 'rb') as handle:
            self.cache = pickle.load(handle)

    def save_to_disk(self):
        with open(self.cache_file, 'wb') as handle:
            pickle.dump(self.cache, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def get_elmo(self, sentence):
        key = str(sentence)
        if key in self.cache:
            return self.cache[key]
        else:
            vectors = elmoize_sentence(sentence, self.elmo)
            self.cache[key] = vectors
            return vectors

def read_ner_data():
    folder = "/Users/daniel/ideaProjects/allennlp/ontonotes/ColumnFormat/"
    train_files = get_files(folder + "Train")
    test_files = get_files(folder + "Test")
    dev_files = get_files(folder + "Dev")

    def get_paragraphs(file):
        paragraphs = []
        tokens = []
        labels = []
        def create_ner_questions(tokensp, labelsp):
            # first extract the labels
            label_indices = []
            label_list = []
            start = -1
            for i, l in enumerate(labelsp):
                if l == "O" and i > 0 and labelsp[i-1] != "O":   # end of a label span
                    assert start != -1
                    label_indices.append((start, i))  # note that the end index is not inclusive
                    label_list.append(labelsp[i-1].split("-")[1])
                    start = -1
                elif l != "O" and start == -1:
                    start = i
                elif l == "O":  # if we are outside, always set the start index to be -1.
                    start = -1
            qas = []
            for i, span in enumerate(label_indices):
                id = label_list[i]
                ans = " ".join(tokensp[span[0]:span[1]])
                ending = tokensp[-1]
                if ending == ".":
                    ending = "?"
                question = " ".join(tokensp[:span[0]] + ["what"] + tokensp[span[1]:-1] + [ending])
                paragraph = " ".join(tokensp)
                char_idx = find_str(paragraph, ans)
                ans = [{"answer_start": char_idx, "text": ans}]
                question = {"answers": ans, "question": question, "id": id}
                qas.append(question)
                paragraphs.append({"context": paragraph, "qas": qas})

        # iterate through sentences
        with open(file) as f:
            content = f.read().splitlines()
            for l in content:
                split = l.split("\t")
                if len(split) > 2:
                    tokens.append(split[5].strip().replace("/.", "."))
                    labels.append(split[0].strip())
                else: # new line
                    # if len(tokens) > 7: # for now, restrict yourself to longer questions
                    create_ner_questions(tokens, labels)
                    tokens = []
                    labels = []
        return paragraphs


    train_paragraphs = []
    for f in train_files:
        train_paragraphs.extend(get_paragraphs(f))

    dev_paragraphs = []
    for f in train_files:
        dev_paragraphs.extend(get_paragraphs(f))

    test_paragraphs = []
    for f in train_files:
        test_paragraphs.extend(get_paragraphs(f))

    train_data =  {"data": [{"paragraphs": train_paragraphs}]}
    test_data = {"data": [{"paragraphs": test_paragraphs}]}
    dev_data = {"data": [{"paragraphs": dev_paragraphs}]}

    with open('ontonotes_questions_ner_train_full.json', 'w', newline='') as f:
        f.write(json.dumps(train_data))
    with open('ontonotes_questions_ner_test_full.json', 'w', newline='') as f:
        f.write(json.dumps(test_data))
    with open('ontonotes_questions_ner_dev_full.json', 'w', newline='') as f:
        f.write(json.dumps(dev_data))

def load_elmo():

    options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

    elmo = Elmo(options_file, weight_file, 2, dropout=0)
    return elmo

def elmoize_sentence(sentences, elmo):
    character_ids = batch_to_ids(sentences)

    embeddings = elmo(character_ids)

    elmo_representations = embeddings['elmo_representations']

    # embeddings['elmo_representations'][0]
    vect1 = numpy.mean(embeddings['elmo_representations'][0].data.cpu().numpy(), 1)
    vect2 = numpy.mean(embeddings['elmo_representations'][1].data.cpu().numpy(), 1)

    return list(vect1[0]) + list(vect1[0])

def test_elmo():
    from allennlp.modules.elmo import Elmo, batch_to_ids

    options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

    elmo = Elmo(options_file, weight_file, 2, dropout=0)

    # use batch_to_ids to convert sentences to character ids
    sentences = [['elrtuelirt', 'tomorrow', 'kjashkjashd', 'First', 'sentence', '.'], ['First', 'sentence', '.']]
    character_ids = batch_to_ids(sentences)

    embeddings = elmo(character_ids)
    # print(embeddings)
    # print(embeddings)
    print(len(embeddings['elmo_representations']))
    print((embeddings['elmo_representations'][0]).shape)

def read_and_save_arc():
    # first read arc sentences (got em from Tushar)

    # 2nd: read the arc analysis

    pass

def read_and_save_lambada():
    pass

def print_output_weight_vector():
    model, dataset_reader = load_model()

    question = "What kind of test succeeded on its first attempt?"
    paragraph = "One time I was writing a unit test, and it succeeded on the first attempt."

    a = solve(question, paragraph, model, dataset_reader, ["unit test"])
    pass

def plot_matrix():
    import numpy
    import matplotlib.pylab as plt
    m = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]


    matrix = numpy.matrix(m)
    fig, ax = plt.subplots()
    im = ax.imshow(numpy.transpose(matrix), aspect='auto')
    p_labels = ['the-iranian', 'iranian-language', 'language-or', 'or-iranic', 'iranic-language', 'language-form', 'form-a', 'a-branch', 'branch-of', 'of-the', 'the-indo-iranian', 'indo-iranian-language', 'language-,', ',-which', 'which-in', 'in-turn', 'turn-be', 'be-a', 'a-branch', 'branch-of', 'of-the', 'the-indo-european', 'indo-european-language', 'language-family', 'family-.', '.-the', 'the-speaker', 'speaker-of', 'of-iranian', 'iranian-language', 'language-be', 'be-know', 'know-as', 'as-iranian', 'iranian-people', 'people-.', '.-historical', 'historical-iranian', 'iranian-language', 'language-be', 'be-group', 'group-in', 'in-three', 'three-stage', 'stage-:', ':-old', 'old-iranian', 'iranian-(', '(-until', 'until-400', '400-bce', 'bce-)', ')-,', ',-middle', 'middle-iranian', 'iranian-(', '(-400', '400-bce', 'bce-–', '–-900', '900-ce', 'ce-)', ')-,', ',-and', 'and-new', 'new-iranian', 'iranian-(', '(-since', 'since-900', '900-ce', 'ce-)', ')-.', '.-of', 'of-the', 'the-old', 'old-iranian', 'iranian-language', 'language-,', ',-the', 'the-better', 'better-understand', 'understand-and', 'and-record', 'record-one', 'one-be', 'be-old', 'old-persian', 'persian-(', '(-a', 'a-language', 'language-of', 'of-achaemenid', 'achaemenid-iran', 'iran-)', ')-and', 'and-avestan', 'avestan-(', '(-the', 'the-language', 'language-of', 'of-the', 'the-avesta', 'avesta-)', ')-.', '.-middle', 'middle-iranian', 'iranian-language', 'language-include', 'include-middle', 'middle-persian', 'persian-(', '(-a', 'a-language', 'language-of', 'of-sassanid', 'sassanid-iran', 'iran-)', ')-,', ',-parthian', 'parthian-,', ',-and', 'and-bactrian', 'bactrian-.', '']
    q_labels = ['what-be', 'be-the', 'the-iranic', 'iranic-language', 'language-a', 'a-subgroup', 'subgroup-of', 'of-?', '']
    ax.set_yticks(range(len(q_labels)))
    ax.set_xticks(range(len(p_labels)))

    ax.set_yticklabels(q_labels)
    ax.set_xticklabels(p_labels)
    # plt.colorbar()
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
             rotation_mode="anchor")
    fig.tight_layout()
    plt.rcParams["figure.figsize"] = (5, 20)
    plt.show()

def reorder_string(str, except_span):
    from ccg_nlpy import remote_pipeline
    pipeline = remote_pipeline.RemotePipeline(server_api="http://macniece.seas.upenn.edu:4002")
    ta = pipeline.doc(str)
    tokens = ta.get_tokens
    char_offsets = ta.get_token_char_offsets

    former_ans = ""
    if except_span != None:
        former_ans = str[except_span[0]:except_span[1]]

    for i in range(1, len(tokens) - 1, 2):

        if i+1 == len(tokens) - 1:  # don't swtich the final word.
            break

        s1 = char_offsets[i][0]
        e1 = char_offsets[i][1]
        s2 = char_offsets[i+1][0]
        e2 = char_offsets[i+1][1]

        char_middle1 = (s1 + e1)/2
        char_middle2 = (s2 + e2) / 2

        if except_span != None and char_middle1 >= except_span[0] and char_middle1 <= except_span[1]:   # of tokens cover the span, don't shuffle it.
            continue

        if except_span != None and char_middle2 >= except_span[0] and char_middle2 <= except_span[1]:   # of tokens cover the span, don't shuffle it.
            continue

        str = str[0:s1] + str[s2:e2] + str[e1:s2] + str[s1:e1] + str[e2:]

    if except_span != None:
        new_ans = str[except_span[0]:except_span[1]]
        # assert new_ans == former_ans, "answers have changed / new_ans: " + new_ans + " / former_ans: " + former_ans
        if new_ans != former_ans:
            return ""

    return str

def shuffle_bidaf_questions():
    dataset_file = "/Users/daniel/ideaProjects/linear-classifier/other/questionSets/squad-dev-v1.1.json"

    # rerordering questions
    # with open(dataset_file) as file:
    #     dataset_json = json.load(file)
    #     dataset = dataset_json['data']
    #     for aid, article in enumerate(dataset):
    #         print("Progress: " + str(100.0 * aid / len(dataset)))
    #         for pid, paragraph in enumerate(article['paragraphs']):
    #             for qid, qa in enumerate(paragraph['qas']):
    #                 dataset_json["data"][aid]['paragraphs'][pid]["qas"][qid]['question'] = reorder_string(qa['question'], None)
    #                 # qa['question'] = reorder_string(qa['question'], None)
    #             # break
    #         # break
    #     with open('squad-dev-v1.1-reordered-questions.json', 'w') as outfile:
    #         json.dump(dataset_json, outfile)

    # reordering paragraphs
    with open(dataset_file) as file:
        dataset_json = json.load(file)
        dataset = dataset_json['data']
        for aid, article in enumerate(dataset):
            print("Progress: " + str(100.0 * aid / len(dataset)))
            new_paragraphs = []
            for pid, paragraph in enumerate(article['paragraphs']):
                # dataset_json["data"][aid]['paragraphs'][pid]["qas"][qid]['question'] = reorder_string(qa['question'], None)
                context = paragraph['context']
                for qid, qa in enumerate(paragraph['qas']):
                    ans = qa["answers"][0]
                    start = ans["answer_start"]
                    end = ans["answer_start"] + len(ans["text"])
                    new_context = reorder_string(context, (start, end))
                    if new_context != "":
                        new_paragraphs.append({"context": new_context, "qas": [qa]})
            dataset_json["data"][aid]['paragraphs'] = new_paragraphs

        with open('squad-dev-v1.1-reordered-paragraphs.json', 'w') as outfile:
            json.dump(dataset_json, outfile)


def sample_shuffle():
    str = "Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24\u201310 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as \"Super Bowl L\"), so that the logo could prominently feature the Arabic numerals 50."
    print(reorder_string(str, (177, 177 + len("Denver Broncos"))))
    print(str)


def load_srl_model():
    from allennlp.models import load_archive
    archive = load_archive("https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz")
    # config = archive.config.duplicate()
    model = archive.model
    model.eval()
    predictor = Predictor.from_archive(archive, 'semantic-role-labeling')
    return predictor

def test_srl():
    predictor = load_srl_model()
    inputs = {
        "sentence": "The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California."
    }
    result = predictor.predict_json(inputs)
    print(result)

def test_chunker():
    from ccg_nlpy import remote_pipeline
    pipeline = remote_pipeline.RemotePipeline(server_api="http://macniece.seas.upenn.edu:4002")
    ta = pipeline.doc("Two partially reusable launch systems were developed, the Space Shuttle and Falcon 9.")
    constituents = ta.get_shallow_parse
    char_offsets = ta.get_token_char_offsets
    print(constituents)
    print(char_offsets)

lowe_case_first_char = lambda s: s[:1].lower() + s[1:] if s else ''

def replace_two_spans(str, span1, span2, char_offsets):
    # first make sure span1 happens before span2
    # print(span1)
    # print(span2)
    # print(char_offsets)
    if span1["start"] < span2["start"]:
        pass # good to go
    else:
        tmp = span1
        span1 = span2
        span2 = tmp

    # make sure the spans don't have overlaps
    assert span2["start"] >= span1["end"], "the spans have a little overlap"

    # create the new string
    s1 = char_offsets[span1["start"]][0]
    e1 = char_offsets[span1["end"]-1][1]
    s2 = char_offsets[span2["start"]][0]
    e2 = char_offsets[span2["end"]-1][1]

    # if the final character is the punctuation, don't nove it
    # print(e2)
    # print(str[e2-1])
    # print(len(str))

    # if e2 >= len(str) and (str[e2-1] == '.' or str[e2-1] == '?' ):
    #     e2 = e2 - 1

    # print(" -> Replace `" + str[s1:e1] + "` with `" + str[s2:e2] + "`")

    # print("S1: `" + str[s1:e1] + "`")
    # print("between: `" + str[e1:s2] + "`")
    # print("S2: `" + str[s2:e2] + "`")

    # print("Begin: `" + str[0:s1] + "`")
    # print("S1: `" + str[s1:e1] + "`")
    # print("between: `" + str[e1:s2] + "`")
    # print("S2: `" + str[s2:e2] + "`")
    # print("End: `" + str[e2:] + "`")
    # print("Before: " + str)
    if s1 == 0:
        str = str[0:s1] + str[s2:e2] + str[e1:s2] + lowe_case_first_char(str[s1:e1]) + str[e2:]
    else:
        str = str[0:s1] + str[s2:e2] + str[e1:s2] + str[s1:e1] + str[e2:]
    # print("After:  " + str)
    # print("-------")
    return str

def shuffle_with_chunker():
    from ccg_nlpy import remote_pipeline
    pipeline = remote_pipeline.RemotePipeline(server_api="http://macniece.seas.upenn.edu:4002")

    def shuffle(str_old, illegal_span = None):
        ta = pipeline.doc(str_old)
        constituents = ta.get_shallow_parse
        char_offsets = ta.get_token_char_offsets
        shuffled_indices = []
        str1 = str_old
        for i, c in enumerate(constituents):
            if (i < len(constituents) - 1) and constituents[i]["label"] == "VP":
                # print(constituents[i-1]["label"] + constituents[i+1]["label"])
                # make sure the cons before and after are NPs
                if constituents[i - 1]["label"] == "NP" and constituents[i + 1]["label"] == "NP" \
                        and (i - 1) not in shuffled_indices and (i + 1) not in shuffled_indices:
                    c_before = constituents[i - 1]
                    c_after = constituents[i + 1]

                    # don't do shuffling if the illegal span has overlaps with the correct span:
                    begin = char_offsets[c_before["start"]][0]
                    end = char_offsets[c_after["end"] - 1][1]
                    if illegal_span == None or (illegal_span != None and ((end <= illegal_span[0]) or (begin >= illegal_span[1]))):
                        shuffled_indices.append(i - 1)
                        shuffled_indices.append(i + 1)
                        # print("--> done")
                        str_old = replace_two_spans(str_old, c_before, c_after, char_offsets)
                    else:
                        pass
                        # print("--> not done")
                        # print("begin: " + str(begin))
                        # print("end: " + str(end))
                        # print("illegal_span: " + str(illegal_span))
                        # print("shuffled_indices: " + str(shuffled_indices))
        if str1 == str_old:
            return ""
        else:
            return str_old

    # shuffle("What religion were the Normans")

    dataset_file = "/Users/daniel/ideaProjects/linear-classifier/other/questionSets/squad-dev-v1.1.json"

    # rerordering questions
    # with open(dataset_file) as file:
    #     dataset_json = json.load(file)
    #     dataset = dataset_json['data']
    #     for aid, article in enumerate(dataset):
    #         print("Progress: " + str(100.0 * aid / len(dataset)))
    #         for pid, paragraph in enumerate(article['paragraphs']):
    #             new_questions = []
    #             for qid, qa in enumerate(paragraph['qas']):
    #                 # print("===============")
    #                 str_old = qa['question']
    #                 # print("Original: " + str_old )
    #                 str_new = shuffle(str_old)
    #                 # print("Shuffled: " + str_new)
    #                 if str_new != "":
    #                     question = {"answers": qa["answers"], "question": str_new, "id": qa["id"]}
    #                     new_questions.append(question)
    #                 # qa['question'] = reorder_string(qa['question'], None)
    #             dataset_json["data"][aid]['paragraphs'][pid]["qas"] = new_questions
    #             # break
    #         # break
    #     with open('squad-dev-v1.1-reordered-questions-chunker.json', 'w') as outfile:
    #         json.dump(dataset_json, outfile)

    # reordering paragraphs
    with open(dataset_file) as file:
        dataset_json = json.load(file)
        dataset = dataset_json['data']
        for aid, article in enumerate(dataset):
            print("Progress: " + str(100.0 * aid / len(dataset)))
            new_paragraphs = []
            for pid, paragraph in enumerate(article['paragraphs']):
                # dataset_json["data"][aid]['paragraphs'][pid]["qas"][qid]['question'] = reorder_string(qa['question'], None)
                context = paragraph['context']
                for qid, qa in enumerate(paragraph['qas']):
                    print("===============")
                    ans = qa["answers"][0]
                    start = ans["answer_start"]
                    end = ans["answer_start"] + len(ans["text"])
                    print("Original: " + context)
                    new_context = shuffle(context, (start, end))
                    print("Shuffled: " + new_context)
                    if new_context != "":
                        new_paragraphs.append({"context": new_context, "qas": [qa]})
            dataset_json["data"][aid]['paragraphs'] = new_paragraphs

        with open('squad-dev-v1.1-reordered-paragraphs-chunker.json', 'w') as outfile:
            json.dump(dataset_json, outfile)

def shuffle_wiith_srl(sentence, srl_predictor):
    pass

def solve_multirc():
    # evaluate the questions with bidaf

    text_file = "/Users/daniel/ideaProjects/allennlp/ipython/multirc/out22.txt"
    tools.solve_squad_questions("/Users/daniel/ideaProjects/allennlp/QA_datasets/mutlirc_questions.json", text_file)

def convert_mctest_to_json():
    directory = "/Users/daniel/ideaProjects/allennlp/QA_datasets/mctest/"
    paragraphs = []
    def read_mctest_file(file):
        with open(directory + file) as f:
            content = f.read().splitlines()
            for line in content:
                qas = []
                rows = line.split("\t")
                id = rows[0]
                context = rows[2].replace("\\newline", "")
                q1 = rows[3]
                q2 = rows[8]
                q3 = rows[13]
                q4 = rows[18]
                ans1str = rows[4:8]
                ans2str = rows[9:13]
                ans3str = rows[14:18]
                ans4str = rows[18:22]

                def add_question(ans_str, q, i):
                    ans1 = []
                    for s in ans_str:
                        idx = 0
                        if s in content:
                            idx = context.index(s)
                        ans1.append({"answer_start": idx, "text": s})

                    q = q.replace("multiple: ", "")
                    q = q.replace("one: ", "")
                    question = {"answers": ans1, "question": q.strip(), "id": id + "-" + str(i)}
                    qas.append(question)

                paragraphs.append({"context": context.strip(), "qas": qas})

                add_question(ans1str, q1, 1)
                add_question(ans2str, q2, 2)
                add_question(ans3str, q3, 3)
                add_question(ans4str, q4, 4)

    read_mctest_file("mc160.dev.tsv")
    read_mctest_file("mc160.test.tsv")
    read_mctest_file("mc160.train.tsv")
    read_mctest_file("mc500.dev.tsv")
    read_mctest_file("mc500.test.tsv")
    read_mctest_file("mc500.train.tsv")

    questions_json = {"data": [{"paragraphs": paragraphs}]}
    with open('/Users/daniel/ideaProjects/allennlp/QA_datasets/mctest.json', 'w', newline='') as f:
        f.write(json.dumps(questions_json))

if __name__ == "__main__":
    # solve_sample_question()
    solve_squad_questions()
    # sample_clustering()
    # cluster_predictions()
    # example_hierarchical_clustering()
    # find_eigen_values()
    # project_adversarials_with_tsne()
    # project_remedia_with_tsne()

    # filter_squad_questions()
    # printQuestionsForTypingTask()
    # processOutputOfMturk()
    # filter_questions_answer_types()
    # read_ner_data()
    # project_ner_with_tsne()
    # ner_few_shot()
    # test_elmo()
    # typing_classifier()
    # read_and_save_arc()
    # print_output_weight_vector()
    # plot_matrix()
    # shuffle_bidaf_questions()
    # test_srl()
    # test_chunker()
    # shuffle_with_chunker()

    ## question annotations experiments
    # load_babi_questions()
    # project_babi_with_tsne()
    # diagonalize()

    convert_mctest_to_json()

