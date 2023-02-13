import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from itertools import combinations
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from utils import ST_functions
from sklearn import svm
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import accuracy_score
import plotly.express as px
from scipy.spatial import distance
import random


def c3e_sl(piSet, SSet, I, alpha):
    N = len(piSet)
    c = len(piSet[0, :])

    # testando

    # piSet = np.array(piSet)
    y = [[1] * c] * N
    y = np.divide(y, c)
    labels = [-1] * N
    # y = pd.DataFrame(y)
    for k in range(0, I):
        for j in range(0, N):
            diffi = np.arange(0, N)
            cond = diffi != j
            t1 = np.array(SSet[j][cond])
            # http://mathesaurus.sourceforge.net/matlab-numpy.html
            # https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
            p1 = (np.transpose(t1 * np.ones([c, 1])) * y[cond, :]).sum(axis=0)
            p2 = sum(t1)
            y[j, :] = (piSet[j, :] + (2 * alpha * p1)) / (1 + 2 * alpha * p2)
            labels[j] = int(np.where(y[j, :] == np.max(y[j, :]))[0])
    return y, labels


def eds(e, d, p, SSet):
    ### entropy

    w = []
    for i in range(1, p+1):

        ed = np.multiply(e,d)

        if i ==1:
            res = ed
            w.append(np.argmax(res))
        else:
            S_ir = SSet[:,w]
            S_ir = np.sum(S_ir, axis=1)
            S_mean = np.divide(S_ir,(i-1))
            res = np.multiply(ed,(1-S_mean))
            w.append(np.argmax(res))

        e[np.argmax(res)] =-1.E12



    '''

    y=y[0]
    e = calc_class_entropy(y)
    candidates = e > np.percentile(e, 75)
    values = np.array(e)[candidates]

    #### density measure - não funciona bem!
    d = calc_density(SSet)
    candidates = d > np.percentile(d, 75)
    values = np.array(d)[candidates]

    ### low density measure
    l = calc_low_density(DistMat)
    candidates = l > np.percentile(l, 75)
    values = np.array(l)[candidates]


    #### silhouette measure
    from sklearn.metrics import silhouette_samples
    sil_test = np.concatenate([train, test])
    clabels = classAnnotation(sil_test)
    sil_values = silhouette_samples(sil_test, clabels[0])
    s = sil_values[len(test) * (-1):]
    candidates = s > np.percentile(s, 25)
    values = np.array(s)[candidates]

    ### ensembles
    el = np.multiply(e, l)
    candidates = el > np.percentile(el, 75)
    values = np.array(el)[candidates]


    sc = 1 - s
    esc = np.multiply(e, sc)
    candidates = esc > np.percentile(esc, 75)
    values = np.array(esc)[candidates]


    '''
    return w


#SSet = caMatriz

def ic(probs, SSet, train, train_labels, test, test_labels):
    y = c3e_sl(probs, SSet, 5, 0.001)
    for k in range(10):

        e = calc_class_entropy(y[0])
        d = calc_density(SSet)
        w = eds(e, d, 5, SSet)
        [train, train_labels, test, test_labels] = increment_training_set(w, train, train_labels, test, test_labels,k)
        probs = svmClassification(train, train_labels, test)
        SSet = reduce_matrix(w, SSet)
        y = c3e_sl(probs[0], SSet, 5, 0.001)
        acc = accuracy_score(test_labels,probs[1])

        print("Iteration " + str(k + 1) + " - Sizes: Training Set " + str(len(train)) + " - Test Set " + str(len(test))
              + " - Acc: " +str(acc))

def clusterEnsemble(data):
    ssfeat_list = ft.features_subset(data.shape[1], 2)
    max_k = int(len(data) ** (1 / 3))  # equal to cubic root # int(math.sqrt(len(apat_iceds_norm)))
    num_init = 5  # 20
    range_n_clusters = list(range(2, max_k))

    silhouette_list = []
    clusterers_list = []
    cluslabels_list = []
    nuclusters_list = []

    matDist = np.array(euclidean_distances(data, data))

    for n_size_ssfeat in range(int(len(ssfeat_list))):

        # Subconjunto de features
        subset_feat = ssfeat_list[n_size_ssfeat]
        X = data[:, subset_feat]

        best_silhouette_avg = -1.0
        best_clusterer = []
        best_cluster_labels = []
        best_num_clusters = -1

        for n_clusters in range_n_clusters:
            for n_init in range(num_init):

                # Initialize the clusterer with n_clusters value and a random generator
                # seed of 10 for reproducibility.
                clusterer = KMeans(n_clusters=n_clusters, init='random')
                cluster_labels = clusterer.fit_predict(X)

                # The silhouette_score gives the average value for all the samples.
                # This gives a perspective into the density and separation of the formed clusters
                silhouette_avg = silhouette_score(X, cluster_labels)

                if (silhouette_avg > best_silhouette_avg):
                    best_silhouette_avg = silhouette_avg
                    best_clusterer = clusterer
                    best_cluster_labels = cluster_labels
                    best_num_clusters = n_clusters

                # print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
                # clusterer_plots(X, cluster_labels, n_clusters, clusterer)

        silhouette_list.append(best_silhouette_avg)
        clusterers_list.append(best_clusterer)
        cluslabels_list.append(best_cluster_labels)  ### vai usar para gera a matriz de similaridades abaixo
        nuclusters_list.append(best_num_clusters)

    ############# CONSENSO ###################
    cluslabels_list = np.array(cluslabels_list)
    caMatrix = np.array([[0] * cluslabels_list.shape[1]] * cluslabels_list.shape[1])

    for i in range(cluslabels_list.shape[
                       0]):  # for (int i = 0; i < cluEnsemble.length; i++) {  ### TAMANHO DA LISTA cluslabels_list
        for j in range(cluslabels_list.shape[
                           1]):  # for (int j = 0; j < data.numInstances(); j++) { ### len(cluslabels_list[0])
            for k in range(cluslabels_list.shape[
                               1]):  # for (int k = 0; k < data.numInstances(); k++) { ### len(cluslabels_list[0])
                if cluslabels_list[i][j] == cluslabels_list[i][
                    k]:  ######## cluslabels_list[i][j] == cluslabels_list[i][k]
                    caMatrix[j][k] += 1
                if i == cluslabels_list.shape[0] - 1:
                    caMatrix[j][k] = caMatrix[j][k] / cluslabels_list.shape[0]  ### TAMANHO DA LISTA cluslabels_list
    # print("Best Silhoutte =", silhouette_list, " Number of Clusters =", nuclusters_list)
    return [silhouette_list, clusterers_list, cluslabels_list, nuclusters_list, caMatrix, matDist]


def remove_class(hidden_class, train, train_labels):
    train_labels.columns = ['Class']
    labeled_data = pd.concat([train, train_labels], axis=1, sort=False)
    labeled_data = labeled_data[labeled_data.Class != hidden_class]

    t = labeled_data.iloc[:, :-1]
    tl = labeled_data.iloc[:, -1:]

    tl.columns = [0]

    return [t, tl]


def increment_training_set(sel_objects, train, train_labels, test, test_labels, iter = 0,save_dir= '.'):
    if len(train[0]) <= 2:
        ft.visualize_data(test, test_labels, sel_objects, iter, save_dir)

    test = pd.DataFrame(test)
    test_labels = pd.DataFrame(test_labels)
    objects = test.iloc[sel_objects, :]
    objects_labels = test_labels.iloc[sel_objects, :]
    # print("Selected Objects Classes: " + str(objects_labels.values.ravel()))
    train = pd.DataFrame(train)
    train_labels = pd.DataFrame(train_labels)
    train.columns = objects.columns
    train_labels.columns = objects_labels.columns
    tr = pd.concat([train, objects], axis=0)
    trl = pd.concat([train_labels, objects_labels], axis=0)
    te = test.drop(test.index[sel_objects])
    tel = test_labels.drop(test_labels.index[sel_objects])
    return [tr.to_numpy(), trl.to_numpy(), te.to_numpy(), tel.to_numpy()]


def reduce_matrix(sel_objects, SSet):
    sim_mat = np.delete(SSet, np.s_[sel_objects], axis=0)
    sim_mat = np.delete(sim_mat, np.s_[sel_objects], axis=1)
    return sim_mat



def calc_density(s):
    h = 5
    d = [0] * s.shape[0]
    for i in range(s.shape[0]):
        d[i] = np.sum(s[i, :][s[i, :].argsort()[h * (-1):]]) / h
    return d


def calc_low_density(d):
    h = 5
    l = [0] * d.shape[0]
    for i in range(d.shape[0]):
        l[i] = np.sum(d[i, :][d[i, :].argsort()[h * (-1):]]) / h
    return l

def svmClassification(train, train_labels, test):
    SVM = svm.SVC(tol=1.5, probability=True)
    SVM.fit(train, train_labels.ravel())
    probs = SVM.predict_proba(test)
    pred = SVM.predict(test)
    # print(np.around(probs,2))
    return [probs, pred]

def calc_class_entropy(p):
    e = [0] * p.shape[0]
    c = len(p[0, :])
    for i in range(p.shape[0]):
        e[i] = - np.sum(p[i, :] * np.log2(p[i, :])) / np.log2(c)
    return e

if __name__ == "__main__":
    ft = ST_functions()

    train_data_path = 'https://raw.githubusercontent.com/Mailson-Silva/Dataset/main/iris2d-train.csv'
    test_data_path = 'https://raw.githubusercontent.com/Mailson-Silva/Dataset/main/iris2d-test.csv'


    class_index = 2
    df_training = pd.read_csv(train_data_path)
    feat_index = list(range(df_training.shape[1]))
    feat_index.remove(class_index)
    train = df_training.iloc[:, feat_index].values
    train_labels = df_training.iloc[:, class_index].values

    df_test = pd.read_csv(test_data_path)
    feat_index = list(range(df_test.shape[1]))
    feat_index.remove(class_index)
    test = df_test.iloc[:, feat_index].values
    test_labels = df_test.iloc[:, class_index].values

    probs = svmClassification(train, train_labels, test)

    [silhouette_list, clusterers_list, cluslabels_list, nuclusters_list, caMatrix, matDist] = clusterEnsemble(test)
    ic(probs[0], caMatrix, train, train_labels, test, test_labels)


