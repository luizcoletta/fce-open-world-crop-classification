import matplotlib.pyplot as plt
import numpy as np


class ST_graphics:

    def __init__(self):
        pass

    # gráfico da evolução do erro associada a cada classe
    # ---------------------------------------------
    def class_error_graph (self, X, errors, name_metrics) :
        class_errors = []
        style = ['ro--', 'ko--', 'bo--']

        for j in errors:
            cl_erro  = np.array(j).transpose()
            class_errors.append(cl_erro.copy())



        for i in range(class_errors[0].shape(0)):
            plt.figure()
            for j in range(len(name_metrics)):

                plt.plot(X[j], class_errors[j][i], style[j], label=name_metrics[j])

            plt.legend()
            plt.xlabel('Iteração', fontsize=15)
            plt.ylabel('Erro', fontsize=15)
            plt.title('Erro da classe ' + str(i), fontsize=15)
            plt.rcParams['xtick.labelsize'] = 13
            plt.rcParams['ytick.labelsize'] = 13

            #plt.savefig('Erro da classe_' + str(i) + '.png')

    # gráfico de acurácia da função self_training
    # ---------------------------------------------

    def accuracy_graph(self, X, Y, name_metrics ):

        #print('SIlhoueta 0:' + str(y_axis_kmeans))
        #print('SIlhoueta 1:' + str(y_axis_kmeans1))
        #print('Entropia:' + str(y_axis_svm))

        style = ['ro--', 'ko--', 'bo--']

        for i in range(len(X)):
            plt.plot(X[i], Y[i], style[i], label=name_metrics[i])  # approach 0

        plt.legend()
        plt.ylabel('Acurácia', fontsize=14)
        plt.xlabel('Iteração', fontsize=14)
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        #plt.savefig('acurácia_do_self_training.png')
