import argparse
import pandas as pd
import os
import numpy as np

if __name__ == "__main__":
    '''
    dataset_list = ['dp_ceratocystis1']
    classifiers = ['SVM', 'KNN', 'SVM']
    sel_list = ['entropia','silhueta0','silhueta0']
    #linguagem = 'pt'
    '''

    parser = argparse.ArgumentParser(description='Implementação de modelo Open-World para aprendizagem de novas ameaças na lavoura')


    parser.add_argument('-datasets', metavar='datasets', action='store', nargs='+', default=['dp_ceratocystis1'],
                        help='Datasets usados nos experimentos')
    parser.add_argument('-classifiers', metavar='models', action='store', nargs='+', default=['SVM'],
                        help='Classificadores usados')
    parser.add_argument('-selection', metavar='selection', action='store', nargs='+', type=str, default=['entropia'],
                        help='Método de seleção da nova classe')
    parser.add_argument('-language', metavar='language', action='store', type=str, default='pt',
                        help="Idioma dos resultados gerados ('pt' ou 'en')")

    args = parser.parse_args()

    dataset_list = args.datasets
    sel_list = args.selection
    linguagem = args.language
    classifiers = args.classifiers


    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'results/')
    dataset_logs_dir = os.path.join(script_dir, 'logs/')
    data = {}

    dt_cl_sel = [] #datasets/classificador/seleção
    runtime = [] #tempo de execução do experimentp
    obj_nc = [] #objetos da nova classe selecionados
    per_nc = [] #desempenho na classe nova
    acc = [] #acurácia
    f1_score = []
    precisao = []
    recall = []

    for dt in dataset_list:

        for i in range(len(classifiers)):
            dt_cl_sel.append(dt+'/'+classifiers[i]+'/'+sel_list[i])

            time = pd.read_csv(dataset_logs_dir+dt+'/'+classifiers[i]+'/runtimes.csv')
            runtime.append(round(time[sel_list[i]].values[0],2))

            data = pd.read_csv(results_dir +dt+'/'+ classifiers[i] + '/graphics_data/barplot_data.csv')
            objs = round(np.sum(5*data[sel_list[i]]))
            obj_nc.append(objs)

            data = pd.read_csv(results_dir + dt + '/' + classifiers[i] + '/graphics_data/error_class_3.csv')
            res = round(100*(1-data[sel_list[i]].values[-1]),2)
            per_nc.append(res)

            data = pd.read_csv(results_dir + dt + '/' + classifiers[i] + '/graphics_data/Acurácia_data.csv')
            res = round(100*data[sel_list[i]].values[-1],2)
            acc.append(res)

            data = pd.read_csv(results_dir + dt + '/' + classifiers[i] + '/graphics_data/F1-score_data.csv')
            res = round(100*data[sel_list[i]].values[-1],2)
            f1_score.append(res)

            data = pd.read_csv(results_dir + dt + '/' + classifiers[i] + '/graphics_data/Precisão_data.csv')
            res = round(100*data[sel_list[i]].values[-1],2)
            precisao.append(res)

            data = pd.read_csv(results_dir + dt + '/' + classifiers[i] + '/graphics_data/Recall_data.csv')
            res = round(100*data[sel_list[i]].values[-1],2)
            recall.append(res)




    data = {'Dataset/classificador/seleção': dt_cl_sel,
            'Tempo de execução (min)': runtime,
            'N° de objetos rotulados da nova classe':obj_nc,
            'Desempenho na nova classe (%)':per_nc,
            'Acurácia (%)':acc,
            'F1-score (%)':f1_score,
            'Precisão (%)':precisao,
            'Recall (%)': recall}

    pd_data = pd.DataFrame(data)
    pd_data.to_csv(results_dir+dataset_list[0].split('_')[0]+'_summary_table.csv', index=False)








