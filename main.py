
from install_req import install_missing_pkg
check_pkg = install_missing_pkg()  # faz a instalação de pacotes faltantes, se houver
import numpy as np
import pandas as pd
from tensorflow import keras
from ST_modules.Variational_Autoencoder import VAE
from sklearn.metrics import accuracy_score
from ST_modules.Algorithms import alghms
from ST_modules.Plot_graphs import ST_graphics
from utils import ST_functions
from statistics import mean
import skimage.io
from skimage.io import imread_collection
import os
from sklearn.metrics import precision_recall_fscore_support
import time
import argparse


def load_dataset(dataset_name, vae, vae_epoch, lat_dim, len_train):

    script_dir = os.path.dirname(__file__)

    # inserir os outros datasets aqui

    if dataset_name == 'vae_ceratocystis20' and vae == True:
        dir_path = 'data/train_images/dataset_eucapytus/dataset-20/*.png'
        images_path = 'data/train_images/dataset_eucapytus/dataset-20'

        files = os.listdir(images_path)
        files = np.sort(files)
        img_labels = []
        for txt in files:
            n = int(txt.split('.')[0][-1])  # obtem a classe da imagem pelo ultimo caracter do nome do arquivo
            img_labels.append(int(n + 1))

        img_labels = np.array(img_labels)
        col_img = imread_collection(dir_path)

        col_img = col_img.concatenate()

        print('\nRunning VAE to generate latent variables...\n')
        vae_model = VAE(col_img, img_labels, epoch=vae_epoch, lat_dim=lat_dim, shape=np.shape(col_img[0]),
                        len_train=len_train)  # len_train --> tamanho do conjunto de treino
        data = vae_model.output

        print('\nVAE has finished!!\n')

        data_dir = os.path.join(script_dir, 'data/' + dataset_name + '_' + str(lat_dim) + 'D')

        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)

        train_path = 'data/' + dataset_name + '_' + str(lat_dim * 2) + 'D' + '/' + dataset_name+ '_'  + str(
            lat_dim) + 'D.csv'
        data.to_csv(train_path, index=False, header=False)

        test_path = ''
        class_index = lat_dim
        join_data = False
        size_batch = int(len(img_labels) * 0.2)
        class2drop = 3


    if dataset_name == 'vae_ceratocystis10' and vae == True:
        dir_path = 'data/train_images/dataset_eucapytus/dataset-10/*.png'
        images_path = 'data/train_images/dataset_eucapytus/dataset-10'

        files = os.listdir(images_path)
        files = np.sort(files)
        img_labels = []
        for txt in files:
            n = int(txt.split('.')[0][-1])  # obtem a classe da imagem pelo ultimo caracter do nome do arquivo
            img_labels.append(int(n + 1))

        img_labels = np.array(img_labels)
        col_img = imread_collection(dir_path)

        col_img = col_img.concatenate()



        print('\nRunning VAE to generate latent variables...\n')
        vae_model = VAE(col_img, img_labels, epoch=vae_epoch, lat_dim=lat_dim, shape=np.shape(col_img[0]),
                        len_train=len_train)  # len_train --> tamanho do conjunto de treino
        data = vae_model.output

        print('\nVAE has finished!!\n')

        data_dir = os.path.join(script_dir, 'data/' + dataset_name + '_' + str(lat_dim) + 'D')

        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)

        train_path = 'data/' + dataset_name + '_' + str(lat_dim * 2) + 'D' + '/' + dataset_name+ '_'  + str(
            lat_dim) + 'D.csv'
        data.to_csv(train_path, index=False, header=False)

        test_path = ''
        class_index = lat_dim
        join_data = False
        size_batch = int(len(img_labels) * 0.2)
        class2drop = 3


    if dataset_name == 'vae_ceratocystis5' and vae == True:
        dir_path = 'data/train_images/dataset_eucapytus/dataset-5/*.png'
        images_path = 'data/train_images/dataset_eucapytus/dataset-5'

        files = os.listdir(images_path)
        files = np.sort(files)
        img_labels = []
        for txt in files:
            n = int(txt.split('.')[0][-1])  # obtem a classe da imagem pelo ultimo caracter do nome do arquivo
            img_labels.append(int(n + 1))

        img_labels = np.array(img_labels)
        col_img = imread_collection(dir_path)

        col_img = col_img.concatenate()

        print('\nRunning VAE to generate latent variables...\n')
        vae_model = VAE(col_img, img_labels, epoch=vae_epoch, lat_dim=lat_dim, shape=np.shape(col_img[0]),
                        len_train=len_train)  # len_train --> tamanho do conjunto de treino
        data = vae_model.output

        print('\nVAE has finished!!\n')

        data_dir = os.path.join(script_dir, 'data/' + dataset_name + '_' + str(lat_dim) + 'D')

        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)

        train_path = 'data/' + dataset_name + '_' + str(lat_dim * 2) + 'D' + '/' + dataset_name+ '_'  + str(
            lat_dim) + 'D.csv'
        data.to_csv(train_path, index=False, header=False)

        test_path = ''
        class_index = lat_dim
        join_data = False
        size_batch = int(len(img_labels) * 0.2)
        class2drop = 3

    if dataset_name == 'vae_ceratocystis2' and vae == True:
        dir_path = 'data/train_images/dataset_eucapytus/dataset-2/*.png'
        images_path = 'data/train_images/dataset_eucapytus/dataset-2'

        files = os.listdir(images_path)
        files = np.sort(files)
        img_labels = []
        for txt in files:
            n = int(txt.split('.')[0][-1])  # obtem a classe da imagem pelo ultimo caracter do nome do arquivo
            img_labels.append(int(n + 1))

        img_labels = np.array(img_labels)
        col_img = imread_collection(dir_path)

        col_img = col_img.concatenate()



        print('\nRunning VAE to generate latent variables...\n')
        vae_model = VAE(col_img, img_labels, epoch=vae_epoch, lat_dim=lat_dim, shape=np.shape(col_img[0]),
                        len_train=len_train)  # len_train --> tamanho do conjunto de treino
        data = vae_model.output

        print('\nVAE has finished!!\n')

        data_dir = os.path.join(script_dir, 'data/' + dataset_name + '_' + str(lat_dim) + 'D')

        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)

        train_path = 'data/' + dataset_name + '_' + str(lat_dim * 2) + 'D' + '/' + dataset_name+ '_'  + str(
            lat_dim) + 'D.csv'
        data.to_csv(train_path, index=False, header=False)

        test_path = ''
        class_index = lat_dim
        join_data = False
        size_batch = int(len(img_labels) * 0.2)
        class2drop = 3

    if dataset_name == 'vae_ceratocystis1' and vae == True:
        dir_path = 'data/train_images/dataset_eucapytus/dataset-1/*.png'
        images_path = 'data/train_images/dataset_eucapytus/dataset-1'

        files = os.listdir(images_path)
        files = np.sort(files)
        img_labels = []
        for txt in files:
            n = int(txt.split('.')[0][-1])  # obtem a classe da imagem pelo ultimo caracter do nome do arquivo
            img_labels.append(int(n + 1))

        img_labels = np.array(img_labels)
        col_img = imread_collection(dir_path)
        col_img = col_img.concatenate()

        print('\nRunning VAE to generate latent variables...\n')
        vae_model = VAE(col_img, img_labels, epoch=vae_epoch, lat_dim=lat_dim, shape=np.shape(col_img[0]),
                        len_train=len_train)  # len_train --> tamanho do conjunto de treino
        data = vae_model.output

        print('\nVAE has finished!!\n')

        data_dir = os.path.join(script_dir, 'data/' + dataset_name + '_' + str(lat_dim) + 'D')

        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)

        train_path = 'data/' + dataset_name + '_' + str(lat_dim) + 'D' + '/' + dataset_name + '_' + str(
            lat_dim) + 'D.csv'
        data.to_csv(train_path, index=False, header=False)

        test_path = ''
        class_index = lat_dim
        join_data = False
        size_batch = int(len(img_labels) * 0.2)
        class2drop = 3

    if dataset_name == 'hc_ceratocystis20' and vae == False:

        train_path = 'https://raw.githubusercontent.com/Mailson-Silva/Eucaliyptus_dataset/main/hc_features/ceratocystis20.csv'
        test_path = ''
        class_index = 8
        join_data = False
        size_batch = int(6543 * 0.2)
        class2drop = 3

    if dataset_name == 'hc_ceratocystis10' and vae == False:

        train_path = 'https://raw.githubusercontent.com/Mailson-Silva/Eucaliyptus_dataset/main/hc_features/ceratocystis10.csv'
        test_path = ''
        class_index = 8
        join_data = False
        size_batch = int(4070 * 0.2)
        class2drop = 3

    if dataset_name == 'hc_ceratocystis5' and vae == False:

        train_path = 'https://raw.githubusercontent.com/Mailson-Silva/Eucaliyptus_dataset/main/hc_features/ceratocystis5.csv'
        test_path = ''
        class_index = 8
        join_data = False
        size_batch = int(2667 * 0.2)
        class2drop = 3

    if dataset_name == 'hc_ceratocystis2' and vae == False:

        train_path = 'https://raw.githubusercontent.com/Mailson-Silva/Eucaliyptus_dataset/main/hc_features/ceratocystis2.csv'
        test_path = ''
        class_index = 8
        join_data = False
        size_batch = int(1183 * 0.2)
        class2drop = 3

    if dataset_name == 'hc_ceratocystis1' and vae == False:

        train_path = 'https://raw.githubusercontent.com/Mailson-Silva/Eucaliyptus_dataset/main/hc_features/ceratocystis1.csv'
        test_path = ''
        class_index = 8
        join_data = False
        size_batch = int(689 * 0.2)
        class2drop = 3

    if dataset_name == 'dp_ceratocystis20' and vae == False:

        train_path = 'https://raw.githubusercontent.com/Mailson-Silva/Eucaliyptus_dataset/main/dp_features/ceratocystis20.csv'
        test_path = ''
        class_index = 8
        join_data = False
        size_batch = int(6543 * 0.2)
        class2drop = 3

    if dataset_name == 'dp_ceratocystis10' and vae == False:

        train_path = 'https://raw.githubusercontent.com/Mailson-Silva/Eucaliyptus_dataset/main/dp_features/ceratocystis10.csv'
        test_path = ''
        class_index = 8
        join_data = False
        size_batch = int(4070 * 0.2)
        class2drop = 3

    if dataset_name == 'dp_ceratocystis5' and vae == False:

        train_path = 'https://raw.githubusercontent.com/Mailson-Silva/Eucaliyptus_dataset/main/dp_features/ceratocystis5.csv'
        test_path = ''
        class_index = 8
        join_data = False
        size_batch = int(2667 * 0.2)
        class2drop = 3

    if dataset_name == 'dp_ceratocystis1' and vae == False:


        train_path = 'https://raw.githubusercontent.com/Mailson-Silva/Eucaliyptus_dataset/main/dp_features/ceratocystis1.csv'
        test_path = ''
        class_index = 8
        join_data = False
        size_batch = int(689*0.2)
        class2drop = 3

    if dataset_name == 'dp_ceratocystis2' and vae == False:

        train_path = 'https://raw.githubusercontent.com/Mailson-Silva/Eucaliyptus_dataset/main/dp_features/ceratocystis2.csv'
        test_path = ''
        class_index = 8
        join_data = False
        size_batch = int(1183 * 0.2)
        class2drop = 3


    if dataset_name == 'mnist' and vae == True:
        (train, train_labels), (test, test_labels) = keras.datasets.mnist.load_data()
        print(type(train))

        features = np.concatenate((train, test), axis=0)
        img_labels = np.concatenate((train_labels, test_labels), axis=0)

        print('\nRunning VAE to generate latent variables...\n')
        vae_model = VAE(features, img_labels, epoch=vae_epoch, lat_dim= lat_dim, shape =(28,28,1),
                        len_train=len_train)  # len_train --> tamanho do conjunto de treino
        data = vae_model.output

        print('\nVAE has finished!!\n')

        data_dir = os.path.join(script_dir, 'data/' + dataset_name+'_VAE_'+str(lat_dim*2)+'D')

        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)

        train_path = 'data/' + dataset_name+'_VAE_'+str(lat_dim*2)+'D' + '/' + 'mnist_VAE_'+str(lat_dim*2)+'D.csv'
        data.to_csv(train_path, index=False)

        test_path = ''
        class_index = (lat_dim)
        join_data = False
        size_batch = int(70000 * 0.2)
        class2drop = 0

    if dataset_name == 'mnist2D' and vae == False:
        train_path = 'https://raw.githubusercontent.com/Mailson-Silva/weka-dataset/main/mnist_train_2d_weka.csv'
        test_path = 'https://raw.githubusercontent.com/Mailson-Silva/weka-dataset/main/mnist_test_2d_weka.csv'

        '''
        train, train_labels, test, test_labels = ft.separate_features_and_labels(train_path,
                                                                                test_path,
                                                                                class_index=2,
                                                                                class2drop= 0)
        '''
        class_index = 3
        join_data = True
        size_batch = int(70000 * 0.2)
        class2drop = 0


    if dataset_name == 'mnist4D' and vae == False:
        train_path= 'https://raw.githubusercontent.com/Mailson-Silva/mnist_lalent_features/main/mnist_train'
        test_path = 'https://raw.githubusercontent.com/Mailson-Silva/mnist_lalent_features/main/mnist_test'

        '''
        class_index = 4
        df_training = pd.read_csv(train_data_path, header=None,
                                  names=['z_mean1', 'z_mean2', 'z_log_var1', 'z_log_var2', 'labels'])

        df_valores = df_training.loc[df_training['labels'] == 0]
        df_training.drop(df_valores.index, inplace=True)

        feat_index = list(range(df_training.shape[1]))
        feat_index.remove(class_index)
        train = df_training.iloc[:, feat_index].values
        train_labels = df_training.iloc[:, class_index].values

        df_test = pd.read_csv(test_data_path, header=None,
                              names=['z_mean1', 'z_mean2', 'z_log_var1', 'z_log_var2', 'labels'])
        feat_index = list(range(df_test.shape[1]))
        feat_index.remove(class_index)
        test = df_test.iloc[:, feat_index].values
        test_labels = df_test.iloc[:, class_index].values
        '''
        class_index = 5
        join_data = True
        size_batch = int(70000 * 0.2)
        class2drop = 0

    if dataset_name == 'iris' and vae == False:

        train_path= 'https://raw.githubusercontent.com/Mailson-Silva/Dataset/main/iris2d-train.csv'
        test_path = 'https://raw.githubusercontent.com/Mailson-Silva/Dataset/main/iris2d-test.csv'


        class_index = 3
        join_data = True
        size_batch = int(150 * 0.2)
        class2drop = -1


    return train_path, test_path, class_index, join_data, size_batch, class2drop



# Algoritmo de self-training
def self_training(iter, model_name, train, train_labels, test, test_labels, metric,
                  list_new_class_labels, results_dir, n_test_class=10,  kmeans_graph=False):

    # cria variáveis para execução do self-training e gera a pasta para armazenar os resultados
    #----------------------------------------------------------------
    x_axis = []
    y_acc = []
    y_precisao = []
    y_recall = []
    y_fscore = []

    erro_das_classes = []
    erro_da_classe_por_rodada = []
    time_classifier = []
    time_metric = []
    prop_por_rodada = []
    curva_sel_por_rodada = []
    ob = 0.5 * np.shape(train)[0]
    #curva_sel = []
    #prop_por_classe = []

    if len(train[0]) == 2:
        results_dir = os.path.join(results_dir, '/'+metric+'_objects_selected')

        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)


    test_original = test
    labels_original = test_labels
    #--------------------------------------------------------------
    print('\n*******************************************')
    print('   Starting Self-training procedure...    ')
    print('   Classifier: ' + str(model_name) + ' - Metric: ' + str(metric) )
    print('*********************************************')


    #retira os objetos da(s) classe(s) nova(s) do conj. de treino para acrescentar depois
    #-----------------------------------------
    all_labels = np.concatenate((train_labels, test_labels), axis=0)
    test,test_labels = ft.sort_testset(test,test_labels)
    new_classes_objs, test,test_labels = ft.draw_new_classes(list_new_class_labels, test, test_labels)

    if (model_name == 'IC_EDS'):
        [silhouette_list, clusterers_list, cluslabels_list, nuclusters_list, SSet, matDist] = ft.clusterEnsemble(test)

    else:
        SSet = []
        #train, train_labels = ft.sel_exemplares(train, train_labels, ob)


    # Modelo incremental

    classifier_results = alghms(model_name, train, train_labels, test, test_labels, metric, results_dir,
                                    n_test_class, 0, kmeans_graph, SSet)

    preds = classifier_results.pred

    probs = classifier_results.probs
    e = classifier_results.e


    acuracia = accuracy_score(test_labels, preds)
    precisao = precision_recall_fscore_support(test_labels, preds, average='weighted')[0]
    recall = precision_recall_fscore_support(test_labels, preds, average='weighted')[1]
    f_score = precision_recall_fscore_support(test_labels, preds, average='weighted')[2]

    for i in np.unique(labels_original):
        erro = ft.class_error(preds, test_labels, i)
        erro_da_classe_por_rodada.append(erro)

    erro_das_classes.append(erro_da_classe_por_rodada.copy())
    erro_da_classe_por_rodada.clear()
    x_axis.append(0)
    y_acc.append(acuracia)
    y_precisao.append(precisao)
    y_recall.append(recall)
    y_fscore.append(f_score)

    print("\nIteration " + str(0) + " (before train set increment) - Sizes: Training Set " + str(len(train)) + " - Test Set " + str(len(test)) +
          " - Accuracy: " + str(round(acuracia,4)) )

    print('Elapsed time (seg) of classifier ' + str(model_name)+' :' + str(round(classifier_results.classifier_time, 4)))

    print('Elapsed time (seg) of metric ' + str(metric)+' :' + str(round(classifier_results.metric_time, 4)))

    time_metric.append(-1)#round(classifier_results.metric_time, 4))
    time_classifier.append(round(classifier_results.classifier_time, 4))


    for k in range(1, iter + 1):  # 11):
        count_nc = 0

        if len(test) > 0:

            print('\nIteraction: ' + str(k) +' | Running train set increment...')

            if (model_name != 'IC_EDS' and (metric != 'random' and metric != 'aleatória')):

                #curva_sel.append(e.copy())
                #graph.plot_metric_sel(e, metric, results_path, k)

                curva_sel_por_rodada.append(e.copy())

            # https://scikit-learn.org/stable/modules/svm.html

            if(metric == 'EDS'):
                w = ft.eds(e[0], e[1], 5, SSet)

            elif(metric == 'random' or metric == 'aleatória'):
                w = np.random.choice(range(0, test_labels.shape[0]), 5, replace=False)

            else:

                df_e = pd.DataFrame(e)
                df_e.sort_values(by=[0], inplace=True, ascending=False)

                # print(df_e)

                # funcao q a partir de e retorna um w que sao os indices dos c mais confiaveis
                posicoes = df_e.index.values
                posicoes = posicoes.tolist()
                p = 5  # 96

                w = posicoes[0:p]  # posicoes[0:p] # índices (posição) dos objetos que serão retirados do conjunto de teste e colocados no conjunto de treino
            # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html


            [train, train_labels, test, test_labels, objects_labels] = ft.increment_training_set(w, train, train_labels, test,
                                                                                 test_labels, k, results_dir)

            test, test_labels = ft.sort_testset(test, test_labels)

            if len(objects_labels) > 0 or objects_labels != []:
                #all_labels = np.concatenate((train_labels, test_labels), axis=0)
                pro = ft.class_proportion_objects(objects_labels, all_labels)

                prop_por_rodada.append(pro.copy())

            #  ft.visualize_data(train, train_labels,[])]


            if len(test) > 0:
                # https://scikit-learn.org/stable/modules/svm.html

                if(model_name=='IC_EDS'):
                    SSet = ft.reduce_matrix(w, SSet)
                #else:
                    # Modelo incremental
                    #train, train_labels = ft.sel_exemplares(train, train_labels, ob)

                # código que faz a inserção da nova classe durante a execução do modelo
                for nc in list_new_class_labels:
                    if nc[0] == k:

                        test = np.concatenate((test, np.squeeze(new_classes_objs[count_nc][0])), axis=0)
                        test_labels = np.concatenate(
                            (test_labels, np.expand_dims(new_classes_objs[count_nc][1], axis=1)))

                        if (model_name == 'IC_EDS'):
                            [silhouette_list, clusterers_list, cluslabels_list, nuclusters_list, SSet,
                             matDist] = ft.clusterEnsemble(test)

                    count_nc += 1


                classifier_results = alghms(model_name, train, train_labels, test, test_labels, metric, results_dir,
                                            n_test_class, k, kmeans_graph,SSet)

                preds = classifier_results.pred
                probs = classifier_results.probs
                e = classifier_results.e

                acuracia = accuracy_score(test_labels, preds)
                precisao = precision_recall_fscore_support(test_labels, preds, average='weighted')[0]
                recall = precision_recall_fscore_support(test_labels, preds, average='weighted')[1]
                f_score = precision_recall_fscore_support(test_labels, preds, average='weighted')[2]

                for i in np.unique(labels_original):
                    erro = ft.class_error(preds, test_labels, i)
                    erro_da_classe_por_rodada.append(erro)

                erro_das_classes.append(erro_da_classe_por_rodada.copy())
                erro_da_classe_por_rodada.clear()
                x_axis.append(k)
                y_acc.append(acuracia)
                y_precisao.append(precisao)
                y_recall.append(recall)
                y_fscore.append(f_score)

            print(
                "\nIteration " + str(k) + " - Sizes: Training Set " + str(len(train)) + " - Test Set " + str(len(test)) +
                " - Accuracy: " + str(round(acuracia,4)))

            print('Elapsed time (seg) of classifier ' + str(model_name) + ' :' + str(
                round(classifier_results.classifier_time, 4)))

            print('Elapsed time (seg) of metric ' + str(metric) + ' :' + str(round(classifier_results.metric_time, 4)) +'\n')

            time_metric.append(round(classifier_results.metric_time, 4))
            time_classifier.append(round(classifier_results.classifier_time, 4))


            # print(pd.crosstab(pd.Series(test_labels.ravel(), name='Real'), pd.Series(preds, name='Predicted'), margins=True))
            # classes = ['wilt', 'rest']
            # print(metrics.classification_report(test_labels, preds, target_names=classes))
        else:
            print('\n Iteraction ' + str(k) + '--> Test set empty, self-training is done!\n')
            time_metric.append(-1)
            time_classifier.append(-1)

    erros = erro_das_classes.copy()
    x = x_axis.copy()
    y = []
    y.append(y_acc.copy())

    y.append(y_precisao.copy())
    y.append(y_recall.copy())
    y.append(y_fscore.copy())

    #y = y_acc.copy()
    prop_por_classe = prop_por_rodada.copy()


    '''
    #Salva arquivo csv na pasta logs contendo o tempo gasto em cada iteração
    steps = list(range(0, iter+1))
    time_data = {'steps': steps, 'classifier_time (s)': time_classifier, 'metric_time (s)': time_metric}
    elapsed_time = pd.DataFrame(time_data)
    elapsed_time.to_csv('logs/'+dataset_name+'/'+model_name+'&'+metric+'_elapsed_time.csv', index=False)
    '''

    return x, y, erros, curva_sel_por_rodada, time_classifier, time_metric, prop_por_classe


def main(dataset_name, model_name, metric, use_vae , vae_epoch, lat_dim, len_train, n_int, list_new_class_labels,
         linguagem, n_test_class=10,  kmeans_graph=False):

    train_path, test_path, class_index, join_data, size_batch, class2drop = load_dataset(dataset_name, use_vae,
                                                                                         vae_epoch=vae_epoch, lat_dim= lat_dim,
                                                                                   len_train=len_train)

    for i in range(len(model_name)):

        results_dir = os.path.join(os.path.dirname(__file__), 'results/' + dataset_name+'/'+model_name[i])

        if not os.path.isdir(results_dir):

            # cria pastas para armazenar os resultados
            script_dir = os.path.dirname(__file__)
            results_dir = os.path.join(script_dir, 'results/' + dataset_name+'/'+model_name[i])

            if not os.path.isdir(results_dir):
                os.makedirs(results_dir)

            dataset_logs_dir = os.path.join(script_dir, 'logs/' + dataset_name+'/'+model_name[i])

            if not os.path.isdir(dataset_logs_dir):
                os.makedirs(dataset_logs_dir)

            data_graphs = os.path.join(script_dir, 'results/' + dataset_name+'/'+model_name[i] +'/graphics_data')

            if not os.path.isdir(data_graphs):
                os.makedirs(data_graphs)


            runtime = {}
            #listas para geração de gráficos
            all_errors = []
            all_x_axis = []
            all_y_axis = []
            all_metrics = []
            all_time_classifier = []
            all_time_metric = []
            all_props = []
            all_curvas_sel = []
            class_errors = []
            x_axis = []
            y_axis = []
            class_proportion = []
            curv_sel = []

            time_per_model = []

        start = time.time()

        for j in range(1,6):
            print('\n*******************************************')
            print("TRAINING SET AND TEST SET - fold " + str(j))

            if dataset_name.split('_')[0] == 'hc':
                train, train_labels, test, test_labels = ft.get_batch_data(train_path, test_path, class_index, join_data, size_batch, j, class2drop, scale = True)

            else:
                train, train_labels, test, test_labels = ft.get_batch_data(train_path, test_path, class_index, join_data, size_batch, j, class2drop)

            x_ent, y_ent, erros_ent,curva_sel, time_classifier, time_metric, prop_por_classe = self_training(n_int, model_name[i], train, train_labels, test,
                                                    test_labels, metric[i], list_new_class_labels, results_dir,  n_test_class,
                                                    kmeans_graph)




            all_errors.append(erros_ent)
            all_x_axis.append(x_ent)
            all_y_axis.append(y_ent)
            all_time_classifier.append(time_classifier)
            all_time_metric.append(time_metric)
            all_props.append(prop_por_classe)

        if curva_sel != []:
            all_curvas_sel.append(curva_sel.copy())

        # Faz a média dos resultados obtidos em cada fold e armazena em uma lista para plotagem em gráficos:

        #print(np.shape(all_curvas_sel[0][-1]))
        #[print(values) for values in zip(*all_errors)]

        '''
        mean_curves = [np.mean(values) for values in zip(*all_curvas_sel)]

        curv_sel.append(mean_curves)
        print(np.shape(mean_curves[0]))
        all_curvas_sel.clear()
        '''
        mean_errors = [np.mean(values,axis=0) for values in zip(*all_errors)]
        #print(np.shape(mean_errors))
        class_errors.append(mean_errors)
        all_errors.clear()

        mean_props = [np.mean(values, axis=0) for values in zip(*all_props)]
        class_proportion.append(mean_props)
        all_props.clear()

        mean_x_axis = [mean(values) for values in zip(*all_x_axis)]

        x_axis.append(mean_x_axis)
        all_x_axis.clear()

        mean_y_axis = [np.mean(values,axis = 0) for values in zip(*all_y_axis)]

        y_axis.append(mean_y_axis)
        all_y_axis.clear()

        all_metrics.append(metric[i])

        mean_time_classifier = [mean(values) for values in zip(*all_time_classifier)]
        all_time_classifier.clear()
        mean_time_classifier = ['' if value == -1 else value for value in mean_time_classifier]

        mean_time_metric = [mean(values) for values in zip(*all_time_metric)]
        all_time_metric.clear()
        mean_time_metric = ['' if value == -1 else value for value in mean_time_metric]

        # Salva arquivo csv na pasta logs contendo o tempo gasto em cada iteração
        steps = list(range(0, n_iter + 1))
        time_data = {'steps': steps, 'classifier_time (s)': mean_time_classifier, 'metric_time (s)': mean_time_metric}
        elapsed_time = pd.DataFrame(time_data)
        elapsed_time.to_csv(dataset_logs_dir + '/' + model_name[i] + '&' + metric[i] + '_elapsed_time.csv', index=False)

        finish = time.time()
        total_time = (finish - start)/60
        time_per_model.append(total_time)


        if i == len(model_name)-1:
            #print(curv_sel)
            idx = indexes(model_name, model_name[i])
            mtc = [metric[x] for x in idx]

            graph.class_error_graph(x_axis, class_errors, all_metrics, test_labels, data_graphs, results_dir, dataset_name, linguagem)

            if model_name[i] != 'IC_EDS':
                graph.plot_metric_sel(all_curvas_sel, all_metrics, results_dir, linguagem)

            graph.accuracy_graph(x_axis, y_axis, all_metrics, data_graphs, results_dir, dataset_name, linguagem)
            graph.accuracy_all_class_graph(mtc, data_graphs, results_dir, test_labels, class_proportion, list_new_class_labels, linguagem)

            for x in range(len(mtc)):
                print('Elapsed time (min) for ' + str(model_name[i]) + ' and ' + str(mtc[x]) + ' : ' + str(
                    time_per_model[x]))
                runtime.update({str(mtc[x]): time_per_model[x]})

            print(runtime)
            runtime_csv = pd.DataFrame(runtime, index= [0])
            runtime_csv.to_csv(dataset_logs_dir + '/' + 'runtimes.csv', index=False)

        elif model_name[i+1]!=model_name[i]:
            # print(curv_sel)
            idx = indexes(model_name, model_name[i])
            mtc = [metric[x] for x in idx]

            graph.class_error_graph(x_axis, class_errors, all_metrics, test_labels, data_graphs, results_dir,
                                    dataset_name, linguagem)

            if model_name[i] != 'IC_EDS':
                graph.plot_metric_sel(all_curvas_sel, all_metrics, results_dir, linguagem)

            graph.accuracy_graph(x_axis, y_axis, all_metrics, data_graphs, results_dir, dataset_name, linguagem)
            graph.accuracy_all_class_graph(mtc, data_graphs, results_dir, test_labels, class_proportion,
                                           list_new_class_labels, linguagem)

            for x in range(len(mtc)):
                print('Elapsed time (min) for ' + str(model_name[i]) + ' and ' + str(mtc[x]) + ' : ' + str(
                    time_per_model[x]))
                runtime.update({str(mtc[x]):time_per_model[x]})

            runtime_csv = pd.DataFrame(runtime, index=[0])
            runtime_csv.to_csv(dataset_logs_dir + '/' +'runtimes.csv', index=False)




def indexes(iterable, obj):
    return list((index for index, elem in enumerate(iterable) if elem == obj))

if __name__ == "__main__":

    #check_pkg = install_missing_pkg()  # faz a instalação de pacotes faltantes, se houver
    ft = ST_functions() # cria objeto contendo funções diversas a serem usadas ao longo do código
    graph = ST_graphics()

    parser = argparse.ArgumentParser(description='Implementação de modelo Open-World para aprendizagem de novas ameaças na lavoura')

    parser.add_argument('-n_classes', metavar='n_classes', action='store', nargs='?', type= int, default = 3, help='N° de classes do conjunto teste')
    parser.add_argument('-dataset', metavar='dataset', action='store', type=str, required=True, help='Nome do dataset')
    parser.add_argument('-use_vae', metavar='use_vae', action='store', type=bool, default= False, help='Solicita ou não uso do VAE')
    parser.add_argument('-vae_epochs', metavar='vae_epochs', action='store', nargs='?', type=int, default= 100, help='Qtd. de épocas do VAE')
    parser.add_argument('-latent_dim', metavar='latent_dim', action='store', nargs='?', type=int, default= 4,  help='Qtd. dimensões latentes do VAE')
    parser.add_argument('-classifiers', metavar='classifiers', action='store', nargs='+', type= str, default=['svm'], help='Classificadores usados no modelo')
    parser.add_argument('-selection', metavar='selection', action='store', nargs='+', type=str, default=['entropia'], help='Método de seleção da nova classe')
    parser.add_argument('-insert_nc', metavar='insert_nc', action='store', nargs='+', type=int, required=True, help='Insere a classe nova especificada na iteração especificada ([iteração rótulo_nc])')
    parser.add_argument('-iteractions', metavar='iteractions', action='store', nargs='?', type=int, default= 10, help='Qtd. de iterações a ser executadas no modelo')
    parser.add_argument('-language', metavar='language', action='store', type=str, default='pt', help="Idioma dos resultados gerados ('pt' ou 'en')")

    args = parser.parse_args()
    '''

    #print(args.n_classes, type(args.n_classes))

    # PARÂMETROS:
    n_test_class = 3
    dataset_name = 'dp_ceratocystis1'
    use_vae = False   # se verdadeiro usa o VAE para reduzir dimensionalidade do dataset
    len_train = 60000   # tamanho do conjunto de treinamento do dataset para uso do VAE
    vae_epochs = 100     # quantidade de épocas para a execução do VAE
    lat_dim = 4        # quantidade de variaveis latentes do VAE
    sel_model = ['svm'] #, 'ic_eds'] # define o classificador a ser usado
    metric = ['entropia'] #, 'ent&dens'] # define a metrica para descobrir classes novas
    #sel_model = ['svm','svm','svm']#,'ic_eds']  # define o classificador a ser usado
    #metric = ['silhueta0', 'silhueta1', 'entropia']
    list_new_class_labels = [[2, 3]]
    n_iter = 10         # numero de iterações da rotina de self-training
    linguagem = 'pt' # idioma dos resultados (pt ou en)

    '''

    # PARÂMETROS:
    n_test_class = args.n_classes
    dataset_name = args.dataset
    use_vae = args.use_vae
    len_train = []
    vae_epochs = args.vae_epochs
    lat_dim = args.latent_dim
    sel_model = args.classifiers
    metric = args.selection
    list_new_class_labels = np.reshape(args.insert_nc, (int(len(args.insert_nc)/2), 2))
    n_iter = args.iteractions
    linguagem = args.language


    main(dataset_name, sel_model, metric, use_vae, vae_epochs, lat_dim, len_train, n_iter, list_new_class_labels, linguagem, n_test_class)


