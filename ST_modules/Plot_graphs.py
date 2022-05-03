import matplotlib.pyplot as plt
import numpy as np
import os
'''
script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, 'Results/')
sample_file_name = "sample"

if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

'''
class ST_graphics:

    def __init__(self):

        pass

    # gráfico da evolução do erro associada a cada classe
    # ---------------------------------------------
    def class_error_graph (self, X, errors, name_metrics, test_labels, results_dir, dataset_name) :
        class_errors = []
        file_name = "/" + dataset_name + "_error_class_"
        style = ['ro--', 'ko--', 'bo--']

        for j in errors:
            cl_erro  = np.array(j).transpose()
            class_errors.append(cl_erro.copy())

        for i in range(class_errors[0].shape[0]):
            plt.figure()
            for j in range(len(name_metrics)):

                plt.plot(X[j], class_errors[j][i], style[j], label=name_metrics[j])

            plt.legend()
            plt.xlabel('Iteração', fontsize=15)
            plt.ylabel('Erro', fontsize=15)
            plt.title('Erro da classe ' + str(np.unique(test_labels)[i]), fontsize=15)
            plt.rcParams['xtick.labelsize'] = 13
            plt.rcParams['ytick.labelsize'] = 13
            #plt.savefig('./teste/Erro da classe_' + str(i) + '.png')
            plt.savefig(results_dir+file_name+str(np.unique(test_labels)[i])+'.png')
            #plt.show()

    # gráfico de acurácia da função self_training
    # ---------------------------------------------

    def accuracy_graph(self, X, Y, name_metrics, results_dir, dataset_name):

        #print('SIlhoueta 0:' + str(y_axis_kmeans))
        #print('SIlhoueta 1:' + str(y_axis_kmeans1))
        #print('Entropia:' + str(y_axis_svm))

        file_name = "/accuracy_"+ dataset_name +".jpg"

        style = ['ro--', 'ko--', 'bo--']

        plt.figure()

        for i in range(len(X)):
            plt.plot(X[i], Y[i], style[i], label=name_metrics[i])  # approach 0

        plt.legend()
        plt.ylabel('Acurácia', fontsize=14)
        plt.xlabel('Iteração', fontsize=14)
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        #plt.savefig('acurácia_do_self_training.png')
        plt.savefig(results_dir+file_name)
        #plt.show()
