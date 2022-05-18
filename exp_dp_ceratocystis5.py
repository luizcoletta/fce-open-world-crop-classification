import numpy as np
import pandas as pd
from install_req import install_missing_pkg
from tensorflow import keras
from ST_modules.Variational_Autoencoder import VAE
from sklearn.metrics import accuracy_score
from ST_modules.Algorithms import alghms
from ST_modules.Plot_graphs import ST_graphics
from utils import ST_functions
from statistics import mean
import os


def load_dataset(dataset_name, vae, vae_epoch, lat_dim, len_train):
    script_dir = os.path.dirname(__file__)

    # inserir os outros datasets aqui

    if dataset_name == 'dp_ceratocystis5' and vae == False:
        '''
        script_dir = os.path.dirname(__file__)
        data_path = os.path.join(script_dir, 'data/'+dataset_name+'_train.csv')

        if os.path.exists(data_path):
            print('Arquivos csv para '+dataset_name+' já existem!')
        else:
            data = 'https://raw.githubusercontent.com/Mailson-Silva/Eucaliyptus_dataset/main/dp_features/ceratocystis1.csv'
            ft.train_and_test_set_generator(data, dataset_name, 3, 8, 0.2)



        train_path = 'data/'+dataset_name+'_train.csv'
        test_path = 'data/'+dataset_name+'_test.csv'
        train, train_labels, test, test_labels = ft.separate_features_and_labels(train_path,
                                                                             test_path,
                                                                             class_index = 8)

        '''

        train_path = 'https://raw.githubusercontent.com/Mailson-Silva/Eucaliyptus_dataset/main/dp_features/ceratocystis5.csv'
        test_path = ''
        class_index = 8
        join_data = False
        size_batch = int(2667 * 0.2)
        class2drop = 3

    if dataset_name == 'mnist' and vae == True:
        (train, train_labels), (test, test_labels) = keras.datasets.mnist.load_data()

        print('\nRunning VAE to generate latent variables...\n')
        vae_model = VAE(train, train_labels, test, test_labels, epoch=vae_epoch, lat_dim=lat_dim,
                        len_train=len_train)  # len_train --> tamanho do conjunto de treino
        data = vae_model.output

        print('\nVAE has finished!!\n')

        data_dir = os.path.join(script_dir, 'data/' + dataset_name + '_VAE_' + str(lat_dim * 2) + 'D')

        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)

        train_path = 'data/' + dataset_name + '_VAE_' + str(lat_dim * 2) + 'D' + '/' + 'mnist_VAE_' + str(
            lat_dim * 2) + 'D.csv'
        data.to_csv(train_path, index=False)

        test_path = ''
        class_index = (lat_dim * 2)
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

        '''
        class_index = 2
        df_training = pd.read_csv(train_data_path)
        df_valores = df_training.loc[df_training['class'] == 0]
        df_training.drop(df_valores.index, inplace=True)

        feat_index = list(range(df_training.shape[1]))
        feat_index.remove(class_index)
        train = df_training.iloc[:, feat_index].values
        train_labels = df_training.iloc[:, class_index].values

        df_test = pd.read_csv(test_data_path)
        feat_index = list(range(df_test.shape[1]))
        feat_index.remove(class_index)
        test = df_test.iloc[:, feat_index].values
        test_labels = df_test.iloc[:, class_index].values
        '''

    if dataset_name == 'mnist4D' and vae == False:
        train_path = 'https://raw.githubusercontent.com/Mailson-Silva/mnist_lalent_features/main/mnist_train'
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
        train_path = 'https://raw.githubusercontent.com/Mailson-Silva/Dataset/main/iris2d-train.csv'
        test_path = 'https://raw.githubusercontent.com/Mailson-Silva/Dataset/main/iris2d-test.csv'

        '''        
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
        '''
        '''
        train, train_labels, test, test_labels = ft.separate_features_and_labels(train_data_path,
                                                                                test_data_path,
                                                                               class_index=2)
        '''

        class_index = 3
        join_data = True
        size_batch = int(150 * 0.2)
        class2drop = -1

    return train_path, test_path, class_index, join_data, size_batch, class2drop


# Algoritmo de self-training

def self_training(iter, model_name, train, train_labels, test, test_labels, metric,
                  n_test_class=10, kmeans_graph=False):
    # cria variáveis para execução do self-training e gera a pasta para armazenar os resultados
    # ----------------------------------------------------------------
    x_axis = []
    y_axis = []
    erro_das_classes = []
    erro_da_classe_por_rodada = []
    time_classifier = []
    time_metric = []

    if len(train[0]) == 2:
        script_dir = os.path.dirname(__file__)
        results_dir = os.path.join(script_dir, 'results/' + dataset_name + '/' + metric + '_objects_selected')

        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
    else:
        results_dir = ''

    test_original = test
    labels_original = test_labels
    # --------------------------------------------------------------
    print('\n*******************************************')
    print('   Starting Self-training procedure...    ')
    print('   Classifier: ' + str(model_name) + ' - Metric: ' + str(metric))
    print('*********************************************')

    classifier_results = alghms(model_name, train, train_labels, test, test_labels, metric, results_dir,
                                n_test_class, 0, kmeans_graph)

    preds = classifier_results.pred
    probs = classifier_results.probs
    e = classifier_results.e

    acuracia = accuracy_score(test_labels, preds)

    for i in np.unique(labels_original):
        erro = ft.class_error(preds, test_labels, i)
        erro_da_classe_por_rodada.append(erro)

    erro_das_classes.append(erro_da_classe_por_rodada.copy())
    erro_da_classe_por_rodada.clear()
    x_axis.append(0)
    y_axis.append(acuracia)

    print("\nIteration " + str(0) + " (before train set increment) - Sizes: Training Set " + str(
        len(train)) + " - Test Set " + str(len(test)) +
          " - Accuracy: " + str(round(acuracia, 4)))

    print('Elapsed time (seg) of classifier ' + str(model_name) + ' :' + str(
        round(classifier_results.classifier_time, 4)))

    print('Elapsed time (seg) of metric ' + str(metric) + ' :' + str(round(classifier_results.metric_time, 4)))

    time_metric.append(-1)  # round(classifier_results.metric_time, 4))
    time_classifier.append(round(classifier_results.classifier_time, 4))

    for k in range(1, iter + 1):  # 11):

        if len(test) > 0:

            print('\nIteraction: ' + str(k) + ' | Running train set increment...')

            # https://scikit-learn.org/stable/modules/svm.html

            df_e = pd.DataFrame(e)
            df_e.sort_values(by=[0], inplace=True, ascending=False)

            # print(df_e)

            # funcao q a partir de e retorna um w que sao os indices dos c mais confiaveis
            posicoes = df_e.index.values
            posicoes = posicoes.tolist()
            p = 5  # 96

            w = posicoes[
                0:p]  # posicoes[0:p] # índices (posição) dos objetos que serão retirados do conjunto de teste e colocados no conjunto de treino

            # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
            # IMPRIMIR A ACURÁCIA

            # print("#------Gráfico de predição-------#")
            # print("#--------------------------------#")
            # [probs, preds] = svmClassification(train, train_labels, test_original)
            # visualize_data(test_original, preds, [])
            # print(str(accuracy_score(labels_original, preds))+"\n") #acurácia da classificação do conjunto de teste original

            [train, train_labels, test, test_labels] = ft.increment_training_set(w, train, train_labels, test,
                                                                                 test_labels, k, results_dir)

            #  ft.visualize_data(train, train_labels,[])
            if len(test) > 0:
                # https://scikit-learn.org/stable/modules/svm.html
                classifier_results = alghms(model_name, train, train_labels, test, test_labels, metric, results_dir,
                                            n_test_class, k, kmeans_graph)

                preds = classifier_results.pred
                probs = classifier_results.probs
                e = classifier_results.e

                acuracia = accuracy_score(test_labels, preds)

                for i in np.unique(labels_original):
                    erro = ft.class_error(preds, test_labels, i)
                    erro_da_classe_por_rodada.append(erro)

                erro_das_classes.append(erro_da_classe_por_rodada.copy())
                erro_da_classe_por_rodada.clear()
                x_axis.append(k)
                y_axis.append(acuracia)

            print(
                "\nIteration " + str(k) + " - Sizes: Training Set " + str(len(train)) + " - Test Set " + str(
                    len(test)) +
                " - Accuracy: " + str(round(acuracia, 4)))

            print('Elapsed time (seg) of classifier ' + str(model_name) + ' :' + str(
                round(classifier_results.classifier_time, 4)))

            print('Elapsed time (seg) of metric ' + str(metric) + ' :' + str(
                round(classifier_results.metric_time, 4)) + '\n')

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
    y = y_axis.copy()

    '''
    #Salva arquivo csv na pasta logs contendo o tempo gasto em cada iteração
    steps = list(range(0, iter+1))
    time_data = {'steps': steps, 'classifier_time (s)': time_classifier, 'metric_time (s)': time_metric}
    elapsed_time = pd.DataFrame(time_data)
    elapsed_time.to_csv('logs/'+dataset_name+'/'+model_name+'&'+metric+'_elapsed_time.csv', index=False)
    '''

    return x, y, erros, time_classifier, time_metric


def main(dataset_name, model_name, metric, use_vae, vae_epoch, lat_dim, len_train, n_int,
         n_test_class=10, kmeans_graph=False):
    train_path, test_path, class_index, join_data, size_batch, class2drop = load_dataset(dataset_name, use_vae,
                                                                                         vae_epoch=vae_epoch,
                                                                                         lat_dim=lat_dim,
                                                                                         len_train=len_train)

    # cria pastas para armazenar os resultados
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'results/' + dataset_name)

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    dataset_logs_dir = os.path.join(script_dir, 'logs/' + dataset_name)

    if not os.path.isdir(dataset_logs_dir):
        os.makedirs(dataset_logs_dir)

    data_graphs = os.path.join(script_dir, 'results/' + dataset_name + '/graphics_data')

    if not os.path.isdir(data_graphs):
        os.makedirs(data_graphs)

    all_errors = []
    all_x_axis = []
    all_y_axis = []
    all_metrics = []
    all_time_classifier = []
    all_time_metric = []

    class_errors = []
    x_axis = []
    y_axis = []

    '''
    # se use_vae é True, então utiliza o VAE para reduzir a dimensionalidade

    if use_vae:
        print('\nRunning VAE to generate latent variables...\n')
        #vae_model = VAE(train, train_labels, test, test_labels, epoch=vae_epoch, lat_dim= lat_dim,
        #                len_train=len_train)  # len_train --> tamanho do conjunto de treino
        #train, train_labels, test, test_labels = vae_model.output

        print('\nVAE has finished!!\n')
    '''

    for i in range(len(model_name)):

        for j in range(1, 6):
            print('\n*******************************************')
            print("TRAINING SET AND TEST SET - fold " + str(j))

            train, train_labels, test, test_labels = ft.get_batch_data(train_path, test_path, class_index, join_data,
                                                                       size_batch, j, class2drop)

            x_ent, y_ent, erros_ent, time_classifier, time_metric = self_training(n_int, model_name[i], train,
                                                                                  train_labels, test,
                                                                                  test_labels, metric[i], n_test_class,
                                                                                  kmeans_graph)

            all_errors.append(erros_ent)
            all_x_axis.append(x_ent)
            all_y_axis.append(y_ent)
            all_time_classifier.append(time_classifier)
            all_time_metric.append(time_metric)

        # Faz a média dos resultados obtidos em cada fold e armazena em uma lista para plotagem em gráficos:

        mean_errors = [np.mean(values, axis=0) for values in zip(*all_errors)]
        class_errors.append(mean_errors)
        all_errors.clear()

        mean_x_axis = [mean(values) for values in zip(*all_x_axis)]
        x_axis.append(mean_x_axis)
        all_x_axis.clear()

        mean_y_axis = [mean(values) for values in zip(*all_y_axis)]
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
        elapsed_time.to_csv('logs/' + dataset_name + '/' + model_name[i] + '&' + metric[i] + '_elapsed_time.csv',
                            index=False)

    graph.class_error_graph(x_axis, class_errors, all_metrics, test_labels, results_dir, dataset_name)
    graph.accuracy_graph(x_axis, y_axis, all_metrics, results_dir, dataset_name)
    graph.accuracy_all_class_graph(metric, results_dir, test_labels)


if __name__ == "__main__":
    check_pkg = install_missing_pkg()  # faz a instalação de pacotes faltantes, se houver
    ft = ST_functions()  # cria objeto contendo funções diversas a serem usadas ao longo do código
    graph = ST_graphics()

    # PARÂMETROS:
    n_test_class = 3
    dataset_name = 'dp_ceratocystis5'
    use_vae = False  # se verdadeiro usa o VAE para reduzir dimensionalidade do dataset
    len_train = 60000  # tamanho do conjunto de treinamento do dataset para uso do VAE
    vae_epochs = 2  # quantidade de épocas para a execução do VAE
    lat_dim = 2  # quantidade de variaveis latentes do VAE
    sel_model = ['svm', 'svm', 'svm']  # define o classificador a ser usado
    metric = ['silhouette0', 'silhouette1', 'entropy']  # define a metrica para descobrir classes novas

    n_iter = 10  # numero de iterações da rotina de self-training

    main(dataset_name, sel_model, metric, use_vae, vae_epochs, lat_dim, len_train, n_iter, n_test_class)

