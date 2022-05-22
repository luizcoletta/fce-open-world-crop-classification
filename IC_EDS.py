import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from itertools import combinations
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
from scipy.spatial import distance
import random


def c3e_sl(self, piSet, SSet, I, alpha):
    N = len(piSet)
    c = len(piSet[0, :])
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


def eds(self, train, test, y, SSet, DistMat):
    ### entropy measuse
    e = self.calc_class_entropy(y)
    candidates = e > np.percentile(e, 75)
    values = np.array(e)[candidates]

    #### density measure - não funciona bem!
    d = self.calc_density(SSet)
    candidates = d > np.percentile(d, 75)
    values = np.array(d)[candidates]

    ### low density measure
    l = self.calc_low_density(DistMat)
    candidates = l > np.percentile(l, 75)
    values = np.array(l)[candidates]

    #### silhouette measure
    from sklearn.metrics import silhouette_samples
    sil_test = np.concatenate([train, test])
    clabels = self.classAnnotation(sil_test)
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

    return [candidates, values]


def ic(self, probs, SSet, train, train_labels, test, test_labels):
    y = self.c3e_sl(probs, SSet, 5, 0.001)
    for k in range(10):
        e = self.calc_class_entropy(y)
        d = self.calc_density(SSet)
        w = self.eds(e, d, 5, SSet)
        [train, train_labels, test, test_labels] = self.increment_training_set(w, train, train_labels, test,
                                                                               test_labels)
        probs = self.svmClassification(train, train_labels, test, test_labels)
        SSet = self.reduce_matrix(w, SSet)
        y = self.c3e_sl(probs, SSet, 5, 0.001)
        print("Iteration " + str(k + 1) + " - Sizes: Training Set " + str(len(train)) + " - Test Set " + str(len(test)))

    def clusterEnsemble(self, data):
        ssfeat_list = self.features_subset(data.shape[1], 2)
        max_k = int(len(data) ** (1 / 3))  # equal to cubic root # int(math.sqrt(len(apat_iceds_norm)))
        num_init = 5  # 20
        range_n_clusters = list(range(2, max_k))

        silhouette_list = []
        clusterers_list = []
        cluslabels_list = []
        nuclusters_list = []

        matDist = np.array(self.euclidean_distances(data, data))

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


    def remove_class(self, hidden_class, train, train_labels):
        train_labels.columns = ['Class']
        labeled_data = pd.concat([train, train_labels], axis=1, sort=False)
        labeled_data = labeled_data[labeled_data.Class != hidden_class]

        t = labeled_data.iloc[:, :-1]
        tl = labeled_data.iloc[:, -1:]

        tl.columns = [0]

        return [t, tl]


    def increment_training_set(self, sel_objects, train, train_labels, test, test_labels, iter, save_dir):
        if len(train[0]) <= 2:
            self.visualize_data(test, test_labels, sel_objects, iter, save_dir)

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


    def reduce_matrix(self, sel_objects, SSet):
        sim_mat = np.delete(SSet, np.s_[sel_objects], axis=0)
        sim_mat = np.delete(sim_mat, np.s_[sel_objects], axis=1)
        return sim_mat



    def calc_density(self, s):
        h = 5
        d = [0] * s.shape[0]
        for i in range(s.shape[0]):
            d[i] = np.sum(s[i, :][s[i, :].argsort()[h * (-1):]]) / h
        return d


    def calc_low_density(self, d):
        h = 5
        l = [0] * d.shape[0]
        for i in range(d.shape[0]):
            l[i] = np.sum(d[i, :][d[i, :].argsort()[h * (-1):]]) / h
        return l
