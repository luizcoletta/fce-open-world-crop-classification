import pandas as pd
from install_req import install_missing_pkg
from tensorflow import keras
from ST_modules.Variational_Autoencoder import VAE
from sklearn.metrics import accuracy_score
from ST_modules.Algorithms import alghms
from ST_modules.Plot_graphs import ST_graphics as graph
from utils import ST_functions as ft

'''
import time
import tensorflow as tf
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from copy import copy
import pandas as pd
from PIL import Image

from keras_segmentation.models.unet import unet, vgg_unet, mobilenet_unet
from keras_segmentation.models.segnet import segnet
from utils import color_list, save_file, roi_extraction, matrix2augimage, iou_metric, create_dir, data_augmentation, data_all


def train_model(model_settings, dataset_name, num_classes, epochs):
    start_time = time.time()
    # Training

    type = model_settings[0]
    height = model_settings[1]
    width = model_settings[2]

    # real image size: 960x720 (ratio -> 4:3), but 720/32 is inexact!
    # type size 1: 960x704, both division by 32 is exact
    # type size 2: 960x736, both division by 32 is exact

    # model = vgg_unet(n_classes=10, input_width=960, input_height=768)
    # model = vgg_unet(n_classes=10, input_width=480, input_height=384)
    # model = vgg_unet(n_classes = num_classes, input_width = width, input_height = height)

    model = []
    if type == "unet":
        model = unet(n_classes=num_classes, input_height=height, input_width=width)
    else:
        if type == "vgg_unet":
            model = vgg_unet(n_classes=num_classes, input_height=height, input_width=width)
        else:
            if type == "mobilenet_unet":
                model = mobilenet_unet(n_classes=num_classes, input_height=height, input_width=width)
            else:
                if type == "segnet":
                    model = segnet(n_classes=num_classes, input_height=height, input_width=width)

    model.train(
        train_images="data/train_images/" + dataset_name + "/images_train/",
        train_annotations="data/train_images/" + dataset_name + "/annotations_train/",
        checkpoints_path="models/" + dataset_name + "/" + type + "/" + type + "_1", epochs=epochs)

    print("--- %s seconds ---" % (time.time() - start_time))

    return model


def load_trained_model(settings, num_classes):

    type = settings[0]
    height = settings[1]
    width = settings[2]

    model = []
    if type == "unet":
        model = unet(n_classes=num_classes, input_height=height, input_width=width)
    else:
        if type == "vgg_unet":
            model = vgg_unet(n_classes=num_classes, input_height=height, input_width=width)
        else:
            if type == "mobilenet_unet":
                model = mobilenet_unet(n_classes=num_classes, input_height=height, input_width=width)
            else:
                if type == "segnet":
                    model = segnet(n_classes=num_classes, input_height=height, input_width=width)
    # PSPNET
    # assert input_height%192 == 0
    # assert input_width%192 == 0

    # https://www.tensorflow.org/tutorials/keras/save_and_load
    latest = tf.train.latest_checkpoint('models/' + type + '/')
    model.load_weights(latest)

    return model


def prediction(model, result_folder, num_classes, test_img_path, test_ann_path):

    img_id_list = []
    iou_classes_list = []
    iou_classes_summary_list = []
    iou_object_list = []
    iou_object_summary_list = []

    create_dir('results/images')
    create_dir('results/annotations')
    create_dir('results/labels')

    files_list = [f for f in listdir(test_img_path) if isfile(join(test_img_path, f))]

    ### https://divamgupta.com/image-segmentation/2019/06/06/deep-learning-semantic-segmentation-keras.html
    '''

'''
from keras_segmentation.predict import predict, predict_multiple
'''

'''
predict(
    	checkpoints_path="checkpoints/vgg_unet_1",
    	inp="dataset_path/images_prepped_test/0016E5_07965.png",
    	out_fname="output.png"
  )'''

'''predict_multiple(
    	checkpoints_path="models/vgg_unet/vgg_unet_1",
    	inp_dir=test_img_path,
    	out_dir="outputs/",
        colors=cc
    )'''

'''
    for f in files_list:

        test_img = test_img_path + f
        test_gt = test_ann_path + f
        output_path = "results/labels/" + result_folder + "_classes_" + f

        out = model.predict_segmentation( ## o tamanho de out é a dimensão da entrada / 2
            inp=test_img, colors=color_list(num_classes),
            out_fname=output_path,
        )

        tested_img = cv2.imread(test_img)
        img_height = tested_img.shape[0]
        img_width = tested_img.shape[1]

        img_res = matrix2augimage(copy(out), (img_width, img_height))
        img_res.save("results/annotations/" + result_folder + "_threshold_" + f[:-4] + ".png")

        img_array = np.array(img_res)
        img_matrix = img_array[:, :, 0]
        img_matrix[img_matrix == 0] = 1
        img_matrix[img_matrix >= 240] = 0
        img_matrix[img_matrix > 0] = 1
        #save_file("results/typification/", "img_matrix.csv", "csv", img_matrix, '%d')
        [roi_img, mask] = roi_extraction(tested_img, img_matrix, [1])
        save_file("results/images/", result_folder + "_object_" + f[:-4], "png", roi_img, [])

        # IoU by classes
        gt_reduced = cv2.resize(cv2.imread(test_gt, 0), (out.shape[1], out.shape[0]))
        iou_class = iou_metric(gt_reduced, out, num_classes)

        # IoU by object
        gt = cv2.imread(test_gt, 0)
        gt[gt > 0] = 1
        iou_object = iou_metric(gt, img_matrix, 2)

        img_id_list.append(f[:-4])
        iou_classes_list.append(iou_class[0])
        iou_classes_summary_list.append(iou_class[1])
        iou_object_list.append(iou_object[0])
        iou_object_summary_list.append(iou_object[1])

    df_results = pd.DataFrame({'IMG_ID': img_id_list})
    df_results = pd.concat([df_results, pd.DataFrame(iou_classes_list)], axis=1)
    df_results = pd.concat([df_results, pd.DataFrame(iou_classes_summary_list)], axis=1)
    df_results = pd.concat([df_results, pd.DataFrame(iou_object_list)], axis=1)
    df_results = pd.concat([df_results, pd.DataFrame(iou_object_summary_list)], axis=1)
    df_results.columns = ['IMG_ID', 'Class0', 'Class1', 'Class2', 'Class3', 'Class4', 'Class5',
                          'Class_Mean', 'Class_Var', 'Class_Std', 'Background', 'Object',
                          'Obj_Mean', 'Obj_Var', 'Obj_Std']
    return df_results


def main(num_classes, dataset_name, validation_task, use_trained_model, epochs, sel_model):

    models_list = [("unet", 192, 160),              # 0
                   ("vgg_unet", 192, 160),          # 1
                   ("mobilenet_unet", 192, 160),    # 2
                   ("segnet", 192, 160)]            # 3

    data_all('data/train_images/dataset_panicum/images_train/',       ###caminho imagem
             'data/train_images/dataset_panicum/annotations_train/',  ###caminho  anotação
             "1.png", '1_teste')
    ###nome para abrir   ###nome para salvar

            



    '''
'''
    if use_trained_model:
        model = load_trained_model(models_list[sel_model], num_classes)
    else:
        model = train_model(models_list[sel_model], dataset_name, num_classes, epochs)

    if validation_task:
        test_img_path = "data/val_images/" + dataset_name + "/images_test/"
        test_ann_path = "data/val_images/" + dataset_name + "/annotations_test/"
    else:
        test_img_path = "data/test_images/"
        test_ann_path = ""

    results = prediction(model, models_list[sel_model][0], num_classes, test_img_path, test_ann_path)

    results.to_csv("results/results_" + models_list[sel_model][0] + ".csv", index=False, header=True)'''

'''
'''


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


# Algoritmo de self-training para o MNIST

def self_training(iter, model_name, train, train_labels, test, test_labels, metric,
                  n_train_class=9, n_test_class=10, kmeans_iter=None, kmeans_graph=False):
    # x_axis.clear()
    # y_axis.clear()
    # erro_das_classes.clear()
    # erro_da_classe_por_rodada.clear()

    x_axis = []
    y_axis = []
    erro_das_classes = []
    erro_da_classe_por_rodada = []

    test_original = test
    labels_original = test_labels

    # [0.1, 0.1, 0.8]         [3]

    # treinar um classificador com o train, train_labels
    # e testar a classificação dele no 'test'
    # vamos usar o probs para saber quais objetos nós vamos levar para training set
    # saida é: probs e preds

    for k in range(1, iter + 1):  # 11):

        if len(test) > 0:

            # https://scikit-learn.org/stable/modules/svm.html
            classifier_results = alghms(model_name, train, train_labels, test, test_labels, metric,
                                        n_train_class, n_test_class, kmeans_iter, kmeans_graph)

            preds = classifier_results.pred, probs = classifier_results.probs, e = classifier_results.e

            acuracia = accuracy_score(test_labels, preds)
            erro_das_classes.append(erro_da_classe_por_rodada.copy())
            erro_da_classe_por_rodada.clear()
            x_axis.append(k)
            y_axis.append(acuracia)

            for i in range(n_test_class):
                erro = ft.class_error(preds, test_labels, i)
                erro_da_classe_por_rodada.append(erro)

            df_e = pd.DataFrame(e)
            df_e.sort_values(by=[0], inplace=True, ascending=False)

            # print(df_e)

            # funcao q a partir de e retorna um w que sao os indices dos c mais confiaveis
            posicoes = df_e.index.values
            posicoes = posicoes.tolist()
            p = 15  # 96

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
                                                                                 test_labels)

            #  ft.visualize_data(train, train_labels,[])

            # https://scikit-learn.org/stable/modules/svm.html

            print(
                "Iteration " + str(k) + " - Sizes: Training Set " + str(len(train)) + " - Test Set " + str(len(test)) +
                "Accuracy: " + str(acuracia) + '\n')

            # print(pd.crosstab(pd.Series(test_labels.ravel(), name='Real'), pd.Series(preds, name='Predicted'), margins=True))
            # classes = ['wilt', 'rest']
            # print(metrics.classification_report(test_labels, preds, target_names=classes))

    erros = erro_das_classes.copy()
    x = x_axis.copy()
    y = y_axis.copy()

    return x, y, erros


def main(dataset_name, model_name, metric, use_vae , vae_epoch, len_train, n_int,
         n_train_class=9, n_test_class=10, kmeans_iter=None, kmeans_graph=False):

    train, train_labels, test, test_labels = load_dataset(dataset_name, use_vae)

    # se use_vae é True, então utiliza o VAE para reduzir a dimensionalidade

    if use_vae:
        print('\nRunning VAE to generate latent variables...\n')
        vae_model = VAE(train, train_labels, test, test_labels, epoch=vae_epoch,
                        len_train=len_train)  # len_train --> tamanho do conjunto de treino
        train, train_labels, test, test_labels = vae_model.output

        print('\nVAE has finished!!\n')

    x_ent, y_ent, erros_ent = self_training(n_int, model_name, train, train_labels, test,
                                            test_labels, metric, n_train_class, n_test_class,
                                            kmeans_iter, kmeans_graph)

    graph.class_error_graph(x_ent, erros_ent, metric)
    graph.accuracy_graph(x_ent, y_ent, metric)

if __name__ == "__main__":

    check_pkg = install_missing_pkg()  # faz a instalação de pacotes faltantes, se houver
    #ft = ST_functions() # cria objeto contendo funções diversas a serem usadas ao longo do código

    # PARÂMETROS:
    num_classes = 2
    dataset_name = 'iris'
    use_vae = False     # se verdadeiro usa o VAE para reduzir dimensionalidade do dataset
    len_train = 60000   # tamanho do conjunto de treinamento do dataset para uso do VAE
    vae_epochs = 2      # quantidade de épocas para a execução do VAE
    sel_model = 'svm'   # define o classificador a ser usado
    metric = 'entropy'  # define a metrica para descobrir classes novas
    n_iter = 8          # numero de iterações da rotina de self-training

    # main(num_classes, dataset_name, validation_task, use_trained_model, epochs, sel_model)
    main(dataset_name, sel_model, metric, use_vae , vae_epochs, len_train, n_iter,
         num_classes, n_test_class=3, kmeans_iter=None, kmeans_graph=False)
