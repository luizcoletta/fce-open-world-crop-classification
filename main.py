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

from install_req import install_missing_pkg
from tensorflow import keras
from ST_modules.Variational_Autoencoder import VAE
from utils import ST_functions as functions
from ST_modules.Algorithms import alghms

def load_dataset(dataset_name):
    #inserir os outros datasets aqui
    if dataset_name == 'mnist':
        (train, train_labels), (test, test_labels) = keras.datasets.mnist.load_data()

    return train, train_labels, test, test_labels


def main(dataset_name, len_train, vae_epoch):
    train, train_labels, test, test_labels = load_dataset(dataset_name)

    vae = True # se True, então utiliza o VAE para reduzir a dimensionalidade

    if vae:
        print('\nRunning VAE to generate latent variables...\n')
        vae_model = VAE(train, train_labels, test, test_labels, epoch=vae_epoch,
                        len_train=len_train)  # len_train --> tamanho do conjunto de treino
        data_train, data_test = vae_model.output

        print('\nVAE has finished!!\n')


if __name__ == "__main__":
    check_pkg = install_missing_pkg()  # faz a instalação de pacotes faltantes, se houver
    fs = functions() # cria objeto contendo funções diversas a serem usadas ao longo do código

    num_classes = 6
    dataset_name = 'mnist'
    len_train = 60000  # tamanho do conjunto de treinamento do dataset
    validation_task = True
    use_trained_model = False
    vae_epochs = 2 #quantidade de épocas para a execução do VAE
    sel_model = 0




    # main(num_classes, dataset_name, validation_task, use_trained_model, epochs, sel_model)
    main(dataset_name, len_train, vae_epochs)
