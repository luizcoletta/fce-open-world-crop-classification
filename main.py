import numpy as np
import pandas as pd
from install_req import install_missing_pkg
from tensorflow import keras
from ST_modules.Variational_Autoencoder import VAE
from sklearn.metrics import accuracy_score
from ST_modules.Algorithms import alghms
from ST_modules.Plot_graphs import ST_graphics
from utils import ST_functions
import os


def load_dataset(dataset_name, vae):
    # inserir os outros datasets aqui

    if dataset_name == 'mnist' and vae == True:
        (train, train_labels), (test, test_labels) = keras.datasets.mnist.load_data()

    if dataset_name == 'mnist' and vae == False:
        train_data_path = 'https://raw.githubusercontent.com/Mailson-Silva/mnist_lalent_features/main/mnist_train'
        test_data_path = 'https://raw.githubusercontent.com/Mailson-Silva/mnist_lalent_features/main/mnist_test'

        class_index = 4
        df_training = pd.read_csv(train_data_path, header=None,
                                  names=['z_mean1', 'z_mean2', 'z_log_var1', 'z_log_var2', 'labels'])
        # df_training.sort_values(by=[4],inplace = 'True',Ascending = 'True')
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

    if dataset_name == 'iris' and vae == False:

        train_data_path= 'https://raw.githubusercontent.com/Mailson-Silva/Dataset/main/iris2d-train.csv'
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

    return train, train_labels, test, test_labels


# Algoritmo de self-training

def self_training(iter, model_name, train, train_labels, test, test_labels, metric,
                  n_test_class=10, kmeans_graph=False):

    # cria variáveis para execução do self-training e gera a pasta para armazenar os resultados
    #----------------------------------------------------------------
    x_axis = []
    y_axis = []
    erro_das_classes = []
    erro_da_classe_por_rodada = []

    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'results/' + dataset_name+ '/'+metric+'_objects_selected')

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    test_original = test
    labels_original = test_labels
    #--------------------------------------------------------------

    print('\nStarting Self-training procedure...')
    print('Classifier: ' + str(model_name) + ' - Metric: ' + str(metric) )

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

    print("\nIteration " + str(0) + " (before train set increment) - Sizes: Training Set " + str(len(train)) + " - Test Set " + str(len(test)) +
          " - Accuracy: " + str(acuracia) )

    print('Elapsed time (seg) of classifier ' + str(model_name)+' :' + str(round(classifier_results.classifier_time, 4)))

    print('Elapsed time (seg) of metric ' + str(metric)+' :' + str(round(classifier_results.metric_time, 4)))



    for k in range(1, iter + 1):  # 11):

        if len(test) > 0:

            print('\nIteraction: ' + str(k) +' | Running train set increment...')

            # https://scikit-learn.org/stable/modules/svm.html


            df_e = pd.DataFrame(e)
            df_e.sort_values(by=[0], inplace=True, ascending=False)

            # print(df_e)

            # funcao q a partir de e retorna um w que sao os indices dos c mais confiaveis
            posicoes = df_e.index.values
            posicoes = posicoes.tolist()
            p = 15  # 96

            w = posicoes[0:p]  # posicoes[0:p] # índices (posição) dos objetos que serão retirados do conjunto de teste e colocados no conjunto de treino

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
                "\nIteration " + str(k) + " - Sizes: Training Set " + str(len(train)) + " - Test Set " + str(len(test)) +
                " - Accuracy: " + str(acuracia) )

            print('Elapsed time (seg) of classifier ' + str(model_name) + ' :' + str(
                round(classifier_results.classifier_time, 4)))

            print('Elapsed time (seg) of metric ' + str(metric) + ' :' + str(round(classifier_results.metric_time, 4)) +'\n')



            # print(pd.crosstab(pd.Series(test_labels.ravel(), name='Real'), pd.Series(preds, name='Predicted'), margins=True))
            # classes = ['wilt', 'rest']
            # print(metrics.classification_report(test_labels, preds, target_names=classes))
        else:
            print('\n Iteraction ' + str(k) + '--> Test set empty, self-training is done!\n')

    erros = erro_das_classes.copy()
    x = x_axis.copy()
    y = y_axis.copy()

    return x, y, erros


def main(dataset_name, model_name, metric, use_vae , vae_epoch, lat_dim, len_train, n_int,
         n_test_class=10, kmeans_graph=False):

    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'results/'+dataset_name)

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    train, train_labels, test, test_labels = load_dataset(dataset_name, use_vae)

    all_errors = []
    all_x_axis = []
    all_y_axis = []
    all_metrics = []

    # se use_vae é True, então utiliza o VAE para reduzir a dimensionalidade

    if use_vae:
        print('\nRunning VAE to generate latent variables...\n')
        vae_model = VAE(train, train_labels, test, test_labels, epoch=vae_epoch, lat_dim= lat_dim,
                        len_train=len_train)  # len_train --> tamanho do conjunto de treino
        train, train_labels, test, test_labels = vae_model.output

        print('\nVAE has finished!!\n')

    for i in range(len(model_name)):
        x_ent, y_ent, erros_ent = self_training(n_int, model_name[i], train, train_labels, test,
                                                test_labels, metric[i], n_test_class,
                                                kmeans_graph)


        all_errors.append(erros_ent)
        all_x_axis.append(x_ent)
        all_y_axis.append(y_ent)
        all_metrics.append(metric[i])

    graph.class_error_graph(all_x_axis, all_errors, all_metrics, test_labels, results_dir, dataset_name)
    graph.accuracy_graph(all_x_axis, all_y_axis, all_metrics, results_dir, dataset_name)

if __name__ == "__main__":

    check_pkg = install_missing_pkg()  # faz a instalação de pacotes faltantes, se houver
    ft = ST_functions() # cria objeto contendo funções diversas a serem usadas ao longo do código
    graph = ST_graphics()

    # PARÂMETROS:
    n_test_class = 3
    dataset_name = 'iris'
    use_vae = False     # se verdadeiro usa o VAE para reduzir dimensionalidade do dataset
    len_train = 60000   # tamanho do conjunto de treinamento do dataset para uso do VAE
    vae_epochs = 2      # quantidade de épocas para a execução do VAE
    lat_dim = 2         # quantidade de variaveis latentes do VAE
    #sel_model = ['svm','svm','svm']  # define o classificador a ser usado
    #metric = ['silhouette0', 'silhouette1', 'entropy']  # define a metrica para descobrir classes novas
    sel_model = ['svm']  # define o classificador a ser usado
    metric = ['silhouette0']  # define a metrica para descobrir classes novas
    n_iter = 7          # numero de iterações da rotina de self-training

    main(dataset_name, sel_model, metric, use_vae , vae_epochs, lat_dim, len_train, n_iter,
         n_test_class, kmeans_graph=True)
