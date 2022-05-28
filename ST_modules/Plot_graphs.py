import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

    def accuracy_all_class_graph(self, metrics, results_dir, test_labels, class_proportion):

        w = 0


        for k in metrics:
            z = 0
            plt.figure(figsize=(10,6))


            cp = class_proportion[w]

            cp = np.transpose(cp)


            for i in np.unique(test_labels):
                data = pd.read_csv(results_dir+'/graphics_data/error_class_'+str(i)+'.csv')

                iter = list(data.iloc[:, 0])
                acc = np.ones(len(iter)) - np.array(data.loc[:, k])
                #prop_bars = np.concatenate((np.array([0]),cp[z]))

                if i == np.unique(test_labels)[-1]:

                    prop_bars = np.concatenate((np.array([0]), cp[int(i)-1]))
                    bars_iter = np.array(iter)

                    plt.bar(bars_iter, prop_bars, ec= 'k',  color = 'green', alpha = 0.3, hatch= '//', width=0.3)
                '''
                if z == 0:
                    bars_iter = np.array(iter) - 0.3
                if z == 1:
                    bars_iter = np.array(iter)
                if z == 2:
                    bars_iter = np.array(iter) + 0.3
                '''

                plt.plot(iter, acc, 'o--', label='class '+str(i))

                #z = z+1

            w = w+1
            #plt.bar(iter, [0,1, 1,1,1,1,1,1,1,1,1])
            plt.legend()
            plt.xlabel('Iteração', fontsize=15)
            plt.ylabel('Acurácia e objetos selecionados da classe 3', fontsize=15)
            plt.title('Resultados obtidos para  ' + str(k), fontsize=15)
            plt.rcParams['xtick.labelsize'] = 13
            plt.rcParams['ytick.labelsize'] = 13
            plt.rcParams["figure.figsize"] = [10.00, 3.50]
            plt.xticks(iter)
            # plt.savefig('./teste/Erro da classe_' + str(i) + '.png')
            plt.savefig(results_dir + '/acur_classes_' + str(k) + '.png')
            # plt.show()

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
            plt.figure(figsize=(10,6))
            error_data = {'iter': X[0]}  # usado para criar arquivo csv com pandas

            plt.rcParams['xtick.labelsize'] = 13
            plt.rcParams['ytick.labelsize'] = 13
            plt.xticks(X[0])

            for j in range(len(name_metrics)):

                plt.plot(X[j], class_errors[j][i], style[j], label=name_metrics[j])
                info = {name_metrics[j]: class_errors[j][i]}
                error_data.update(info.copy())

            error_class = pd.DataFrame(error_data)
            error_class.to_csv('results/' + dataset_name + '/graphics_data/' + 'error_class_'+str(np.unique(test_labels)[i])
                               +'.csv', index=False)

            plt.legend()
            plt.xlabel('Iteração', fontsize=15)
            plt.ylabel('Erro', fontsize=15)
            plt.title('Erro da classe ' + str(np.unique(test_labels)[i]), fontsize=15)

            #plt.savefig('./teste/Erro da classe_' + str(i) + '.png')
            plt.savefig(results_dir+file_name+str(np.unique(test_labels)[i])+'.png')
            #plt.show()

    # gráfico de acurácia da função self_training
    # ---------------------------------------------

    def accuracy_graph(self, X, Y, name_metrics, results_dir, dataset_name):

        #print('SIlhoueta 0:' + str(y_axis_kmeans))
        #print('SIlhoueta 1:' + str(y_axis_kmeans1))
        #print('Entropia:' + str(y_axis_svm))

        accuracy_data = {'iter' : X[0]}
        file_name = "/accuracy_"+ dataset_name +".jpg"

        style = ['ro--', 'ko--', 'bo--']

        plt.figure(figsize=(10,6))

        for i in range(len(X)):
            plt.plot(X[i], Y[i], style[i], label=name_metrics[i])  # approach 0
            info = {name_metrics[i] : Y[i]}
            accuracy_data.update(info.copy())

        plt.legend()
        plt.ylabel('Acurácia', fontsize=15)
        plt.xlabel('Iteração', fontsize=15)
        plt.rcParams['xtick.labelsize'] = 13
        plt.rcParams['ytick.labelsize'] = 13
        plt.xticks(X[0])
        #plt.savefig('acurácia_do_self_training.png')
        plt.savefig(results_dir+file_name)
        #plt.show()

        accuracy_data = pd.DataFrame(accuracy_data)
        accuracy_data.to_csv('results/' + dataset_name + '/graphics_data/'+'accuracy_data.csv', index=False)
