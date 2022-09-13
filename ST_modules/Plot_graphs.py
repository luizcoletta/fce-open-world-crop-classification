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

    def plot_metric_sel(self, sel, metrics, results_dir, dataset_name, linguagem):

        def moving_average(a, n=50):  # n=3
            # from https://stackoverflow.com/a/14314054
            ret = np.cumsum(a, dtype=float)
            ret[n:] = ret[n:] - ret[:-n]
            return ret[n - 1:] / n

        avg_window_size = 20  # 2
        for i in range(np.shape(sel)[1]):
            plt.figure()
            for j in range(np.shape(sel)[0]):

                y = moving_average(sel[j][i], n=avg_window_size)
                x = np.array([i for i in range(len(y))]) + avg_window_size


                plt.plot(x, y, label= metrics[j])

            plt.legend()
            if linguagem == 'pt':
                plt.xlabel('Amostras')
                plt.ylabel('Métrica')
            elif linguagem == 'en':
                plt.xlabel('Samples')
                plt.ylabel('Metric')

            #plt.title('Silhueta dos dados de teste')
            print(results_dir)
            plt.savefig(results_dir + '/'+dataset_name+'_curv_sel' +'_iter_'+ str(i+1) + '.png')


    '''
    def plot_metric_sel(self, sel, metrics, results_dir, k):
        def moving_average(a, n=50):  # n=3
            # from https://stackoverflow.com/a/14314054
            ret = np.cumsum(a, dtype=float)
            ret[n:] = ret[n:] - ret[:-n]
            return ret[n - 1:] / n

        avg_window_size = 50  # 2
        y = moving_average(sel, n=avg_window_size)
        x = np.array([i for i in range(len(y))]) + avg_window_size

        plt.figure()
        plt.plot(x, y)
        # plt.vlines(x=980, ymin=min(silhueta), ymax=max(silhueta))
        # plt.vlines(x=70, ymin=min(silhueta), ymax=max(silhueta))
        # plt.plot((35,min(silhueta)),(35,max(silhueta)),scaley=False)
        plt.xlabel('Amostra')
        plt.ylabel('Métrica')
        # plt.title('Silhueta dos dados de teste')
        plt.savefig(results_dir + '/' +metrics + '_iter_' + str(k) + '.png')
    '''
    def accuracy_all_class_graph(self, metrics, results_dir, test_labels, class_proportion, list_new_class_labels, linguagem):

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
                if linguagem == 'en':
                    plt.plot(iter, acc, 'o--', label='class '+str(i))
                elif linguagem == 'pt':
                    plt.plot(iter, acc, 'o--', label='classe ' + str(i))

                for i in list_new_class_labels:
                    plt.vlines(i[0],ymin=0, ymax=1, colors='r',linestyles ='dashed')

                #z = z+1

            w = w+1
            #plt.bar(iter, [0,1, 1,1,1,1,1,1,1,1,1])
            plt.legend()
            if linguagem == 'pt':
                plt.xlabel('Iteração', fontsize=15)
                plt.ylabel('Acurácia e proporção de objetos selecionados da nova classe', fontsize=15)
                plt.title('Resultados obtidos para  ' + str(k), fontsize=15)
            elif linguagem == 'en':
                plt.xlabel('Iteration', fontsize=15)
                plt.ylabel("Accuracy and new class's selected object proportion", fontsize=15)
                plt.title('Results achieved with  ' + str(k), fontsize=15)

            plt.rcParams['xtick.labelsize'] = 13
            plt.rcParams['ytick.labelsize'] = 13
            plt.rcParams["figure.figsize"] = [10.00, 3.50]
            plt.xticks(iter)
            # plt.savefig('./teste/Erro da classe_' + str(i) + '.png')
            plt.savefig(results_dir + '/acur_classes_' + str(k) + '.png')
            # plt.show()

    # gráfico da evolução do erro associada a cada classe
    # ---------------------------------------------
    def class_error_graph (self, X, errors, name_metrics, test_labels, results_dir, dataset_name, linguagem) :
        class_errors = []
        file_name = "/" + dataset_name + "_error_class_"
        style = ['ro--', 'ko--', 'bo--','go--']


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

            plt.ylim([0,1.1])
            plt.legend()
            if linguagem == 'pt':
                plt.xlabel('Iteração', fontsize=15)
                plt.ylabel('Taxa de erro', fontsize=15)
                plt.title('Taxa de erro da classe ' + str(np.unique(test_labels)[i]), fontsize=15)

            elif linguagem =='en':
                plt.xlabel('Iteration', fontsize=15)
                plt.ylabel('Error rate', fontsize=15)
                plt.title('Error rate of class ' + str(np.unique(test_labels)[i]), fontsize=15)

            #plt.savefig('./teste/Erro da classe_' + str(i) + '.png')
            plt.savefig(results_dir+file_name+str(np.unique(test_labels)[i])+'.png')
            #plt.show()

    # gráfico de acurácia da função self_training
    # ---------------------------------------------

    def accuracy_graph(self, X, Y, name_metrics, results_dir, dataset_name, linguagem):


        #print('SIlhoueta 0:' + str(y_axis_kmeans))
        #print('SIlhoueta 1:' + str(y_axis_kmeans1))
        #print('Entropia:' + str(y_axis_svm))




        style = ['ro--', 'ko--', 'bo--','go--']
        ind_en = ['Accuracy', 'Precision', 'Recall', 'F1-score']
        ind_pt = ['Acurácia', 'Precisão', 'Recall', 'F1-score']
        #file_name = "/accuracy_" + dataset_name + ".jpg"

        for x in range(np.shape(Y)[1]):

            _data = {'iter': X[0]}

            plt.figure(figsize=(10,6))

            for i in range(len(X)):
                plt.plot(X[i], Y[i][x], style[i], label=name_metrics[i])  # approach 0
                info = {name_metrics[i] : Y[i][x]}
                _data.update(info.copy())

            plt.legend()
            if linguagem == 'pt':
                plt.ylabel(ind_pt[x], fontsize=15)
                plt.xlabel('Iteração', fontsize=15)
            elif linguagem == 'en':
                plt.ylabel(ind_en[x], fontsize=15)
                plt.xlabel('Iteração', fontsize=15)

            plt.rcParams['xtick.labelsize'] = 13
            plt.rcParams['ytick.labelsize'] = 13
            plt.xticks(X[0])
            #plt.savefig('acurácia_do_self_training.png')

            if linguagem == 'pt':
                plt.savefig(results_dir + '/' + ind_pt[x] + '_' + dataset_name + ".jpg")
                # plt.show()
                pd_data = pd.DataFrame(_data)
                pd_data.to_csv('results/' + dataset_name + '/graphics_data/' + ind_pt[x] + '_data.csv', index=False)

            elif linguagem == 'en':
                plt.savefig(results_dir + '/' + ind_en[x] + '_' + dataset_name + ".jpg")
                # plt.show()
                pd_data = pd.DataFrame(_data)
                pd_data.to_csv('results/' + dataset_name + '/graphics_data/' + ind_en[x] + '_data.csv', index=False)


