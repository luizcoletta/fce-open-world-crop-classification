#inserir a definição dos classificadores aqui

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from sklearn.cluster import KMeans
import numpy as np
from scipy.stats import norm
from scipy.special import logsumexp
import matplotlib.pyplot as plt
from utils import ST_functions
ft = ST_functions()
import time
from .ncm import NCM_classifier
from .deep_nno import DeepNNO


class alghms:

    def __init__(self, model_name, train, train_labels, test, test_labels, metric, results_path,
                 nclusters_test=10, iter_graph = None, kmeans_graph = False, SSet = None, pseudopoints=None
                 ):

        #interaction --> usado para gerar os gráficos a cada iteração
        #graph --> habilita a exibição de gráficos se True
        #nclusters_train e nclusters_test --> aplicados na obtenção do kmeans para cálculo da silhueta
        #SSet -> usado no algoritmo IC_EDS (matriz de similaridade)
        '''
        if model_name == 'KNN':
            start = time.time()

            self.probs, self.pred = self.KNN(train,train_labels, test)

            finish = time.time()
            total_time = finish - start
            self.classifier_time = total_time

        if model_name == 'RF':
            start = time.time()

            self.probs, self.pred = self.RF(train,train_labels, test)

            finish = time.time()
            total_time = finish - start
            self.classifier_time = total_time
        '''

        self.classifier_time = -1
        self.metric_time = -1

        if model_name == 'IC_EDS':
            start = time.time()

            self.probs, self.pred = self.ic_eds(train,train_labels, test,SSet)

            finish = time.time()
            total_time = finish - start
            self.classifier_time = total_time


        if metric =='EDS':
            start = time.time()
            self.e = self.ed(self.probs, SSet)
            finish = time.time()
            total_time = finish - start
            self.metric_time = total_time

        if model_name == 'incremental':
            start = time.time()

            self.pred = ft.nearest_mean_examplars(train, train_labels, test)

            finish = time.time()
            total_time = finish - start
            self.classifier_time = total_time

        if model_name == 'SVM':
            start = time.time()

            self.probs, self.pred = self.svmClassification(train,train_labels, test)

            finish = time.time()
            total_time = finish-start
            self.classifier_time = total_time

        if metric == 'entropy' or metric == 'entropia':
            start = time.time()
            self.e = self.calc_class_entropy(self.probs)
            finish = time.time()
            total_time = finish - start
            self.metric_time = total_time

        if metric == 'silh_mod':
            start = time.time()
            self.e = self.kmeans_for_new_class(train, test, 0, iter_graph, kmeans_graph, results_path,
                                               len(np.unique(train_labels)), nclusters_test)
            finish = time.time()
            total_time = finish - start
            #print(np.unique(train_labels))
            self.metric_time = total_time

        if metric == 'silh_inc':
            start = time.time()
            self.e = ft.kms_for_new_class(pseudopoints, test, 1)

            finish = time.time()
            total_time = finish - start
            self.metric_time = total_time

        if metric == 'silhouette' or metric == 'silhueta':
            start = time.time()
            self.e = self.kmeans_for_new_class(train, test, 1, iter_graph, kmeans_graph, results_path,
                                               len(np.unique(train_labels)), nclusters_test)

            finish = time.time()
            total_time = finish - start
            self.metric_time = total_time

    #random obj selection (baseline)
        if metric == 'random' or metric == 'aleatória':
            start = time.time()
            self.e = []
            finish = time.time()
            total_time = finish - start
            self.metric_time = total_time

    #######BASELINES####
    #########################################
    #------------------------------------------
    #algoritmo IC_EDS (baseline)
    #----------------------------------------------------------------
    def ic_eds(self,train, train_labels, test,SSet):
        probs = self.svmClassification(train, train_labels, test)
        y = ft.c3e_sl(probs[0], SSet, 5, 0.001)

        return [y[0], y[1]]
        #return [probs[0], probs[1]]

    def ed(self, p, SSet):
        e = self.calc_class_entropy(p)
        d = ft.calc_density(SSet)
        return [e,d]
    #----------------------------------------------------------------
 
    # Random Forest classifier
    def RF(self, train, train_labels, test):
        rf = RandomForestClassifier(n_estimators=9, max_depth=5)
        rf.fit(train, train_labels.ravel())
        probs = rf.predict_proba(test)
        pred = rf.predict(test)
        # print(np.around(probs,2))
        return [probs, pred]

    #KNN classifier
    def KNN(self, train, train_labels, test):
        neigh = KNeighborsClassifier(n_neighbors=3)
        neigh.fit(train, train_labels.ravel())
        probs = neigh.predict_proba(test)
        pred = neigh.predict(test)

        # print(np.around(probs,2))
        return [probs, pred]


    #SVM Classifier
    #-----------------------------------------------------------------
    def svmClassification(self, train, train_labels, test):
        SVM = svm.SVC(tol=1.5, probability=True)
        SVM.fit(train, train_labels.ravel())
        probs = SVM.predict_proba(test)
        pred = SVM.predict(test)

        # print(np.around(probs,2))
        return [probs, pred]

    # entropy
    #-------------------------------------------------------------------
    def calc_class_entropy(self, p):
        e = [0] * p.shape[0]
        c = len(p[0, :])
        for i in range(p.shape[0]):
            e[i] = - np.sum(p[i, :] * np.log2(p[i, :])) / np.log2(c)
        return e

    # silhouette
    #-------------------------------------------------------------------
    def kmeans_for_new_class(self, train, test, kmeans_approach, int, graph, results_path,
                             nclusters_train, nclusters_test=3, threshold=0.8):

        kmeans = KMeans(n_clusters=nclusters_train,  # numero de clusters
                        init='k-means++', n_init=10,
                        # método de inicialização dos centróides que permite convergencia mais rápida
                        max_iter=300)  # numero de iterações do algoritmo

        kmeans_test = KMeans(n_clusters=nclusters_test,  # numero de clusters
                             init='k-means++', n_init=10,
                             # método de inicialização dos centróides que permite convergencia mais rápida
                             max_iter=300)  # numero de iterações do algoritmo

        # Visualização do K-means para os dois conjuntos de dados

        pred_train = kmeans.fit_predict(train)
        kmeans_train_center = kmeans.cluster_centers_
        objs_train_to_center_clusters = kmeans.fit_transform(train)  # calcula a distancia de cada ponto até os centros de cada cluster

        pred_test = kmeans_test.fit_predict(test)
        kmeans_test_center = kmeans_test.cluster_centers_
        objs_test_to_center_clusters = kmeans_test.fit_transform(test)

        if len(train[0]) == 2:
            # Visualização do K-means para os dois conjuntos de dados
            # -------------------------------------------------------
            plt.figure()
            plt.scatter(train[:, 0], train[:, 1], c=pred_train)  # posicionamento dos eixos x e y

            # plt.xlim(-75, -30) #range do eixo x
            # plt.ylim(-50, 10) #range do eixo y
            plt.grid()  # função que desenha a grade no nosso gráfico
            plt.scatter(kmeans_train_center[:, 0], kmeans_train_center[:, 1], s=70,
                        c='red')  # posição de cada centroide no gráfico
            plt.title('Conjunto de treinamento')
            plt.savefig(results_path+'/kmeans_train_' + str(int)+'.jpg')
            #print(results_path)
            #plt.savefig('kmeans_train_' + str(int) + '.jpg')
            #plt.show()

            plt.figure()

            plt.scatter(test[:, 0], test[:, 1], c=pred_test)  # posicionamento dos eixos x e y
            # plt.xlim(-75, -30) #range do eixo x
            # plt.ylim(-50, 10) #range do eixo y
            plt.grid()  # função que desenha a grade no nosso gráfico
            plt.scatter(kmeans_test_center[:, 0], kmeans_test_center[:, 1], s=70,
                        c='red')  # posição de cada centroide no gráfico
            plt.title('Conjunto de teste')
            plt.savefig(results_path+'/kmeans_test_' + str(int)+'.jpg')
            #plt.show()

            #------------------------------------------

        data, data_labels, data_centers, data_dists, silhouette = ft.augment_data(2, train, pred_train,
                                                                               kmeans_train_center,
                                                                               objs_train_to_center_clusters, test,
                                                                               pred_test, kmeans_test_center,
                                                                               objs_test_to_center_clusters,
                                                                               kmeans_approach, threshold)

        return silhouette

