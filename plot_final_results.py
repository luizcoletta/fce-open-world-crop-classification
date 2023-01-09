import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse


def plot_overall_datasets_accuracy(datasets_list, sel_list, linguagem, results_dir, ml):

    # Gera gráfico com a acurácia de todos os datasets
    #Um gráfico para cada medida de seleção

    script_dir = os.path.dirname(__file__)
    path_dir = os.path.join(script_dir, 'results/')


    for sel in sel_list:
        plt.figure(figsize=(10,6))
        for dataset in datasets_list:

            plt.rcParams['xtick.labelsize'] = 13
            plt.rcParams['ytick.labelsize'] = 13

            if linguagem == 'pt':
                data = pd.read_csv(os.path.join(path_dir, dataset+'/'+ml+'/graphics_data/Acurácia_data.csv'))
                x = data['iter']
                y = data[sel]
            elif linguagem == 'en':
                data = pd.read_csv(os.path.join(path_dir, dataset + '/'+ml+ '/graphics_data/Accuracy_data.csv'))
                x = data['iter']
                y = data[sel]

            plt.plot(x, y*100, marker='o', linestyle='dashed', label=dataset)


        plt.legend()
        if linguagem == 'pt':
            plt.ylabel('Acurácia (%)', fontsize=15)
            plt.xlabel('Iteração', fontsize=15)
            plt.title('Desempenho usando ' + sel, fontsize=15)

        elif linguagem == 'en':
            plt.ylabel('Accuracy (%)', fontsize=15)
            plt.xlabel('Iteration', fontsize=15)
            plt.title('Performance with ' + sel, fontsize=15)

        plt.savefig(results_dir+sel+'_overall_accuracy.png')

def plot_overall_datasets_new_class_error(datasets_list, sel_list, linguagem, results_dir, ml):

    # Gera gráfico com a acurácia de todos os datasets
    #Um gráfico para cada medida de seleção

    script_dir = os.path.dirname(__file__)
    path_dir = os.path.join(script_dir, 'results/')

    for sel in sel_list:
        plt.figure(figsize=(10,6))
        for dataset in datasets_list:

            plt.rcParams['xtick.labelsize'] = 13
            plt.rcParams['ytick.labelsize'] = 13

            if linguagem == 'pt':
                data = pd.read_csv(os.path.join(path_dir,dataset+ '/'+ml+'/graphics_data/error_class_3.csv'))
                x = data['iter']
                y = data[sel]
            elif linguagem == 'en':
                data = pd.read_csv(os.path.join(path_dir, dataset + '/'+ml+ '/graphics_data/error_class_3.csv'))
                x = data['iter']
                y = data[sel]

            plt.plot(x, y*100, marker='o', linestyle='dashed', label=dataset)


        plt.legend()
        if linguagem == 'pt':
            plt.ylabel('Erro da classe nova (%)', fontsize=15)
            plt.xlabel('Iteração', fontsize=15)
            plt.title('Métrica: ' + sel, fontsize=15)

        elif linguagem == 'en':
            plt.ylabel("New class' error (%)", fontsize=15)
            plt.xlabel('Iteration', fontsize=15)
            plt.title('Metric: ' + sel, fontsize=15)

        plt.savefig(results_dir+sel+'_overall_newclass_error.png')

def plot_overall_datasets_new_class_prop(datasets_list, sel_list, linguagem, results_dir, ml):

    # Gera gráfico com a acurácia de todos os datasets
    #Um gráfico para cada medida de seleção

    script_dir = os.path.dirname(__file__)
    path_dir = os.path.join(script_dir, 'results/')


    for sel in sel_list:
        plt.figure(figsize=(10,6))
        mul = 0
        for dataset in datasets_list:

            width = 0.15
            plt.rcParams['xtick.labelsize'] = 13
            plt.rcParams['ytick.labelsize'] = 13

            if linguagem == 'pt':
                data = pd.read_csv(os.path.join(path_dir,dataset+'/'+ml+'/graphics_data/barplot_data.csv'))
                x = data['iter'][1:]
                y = data[sel][1:]
            elif linguagem == 'en':
                data = pd.read_csv(os.path.join(path_dir, dataset + '/'+ml+'/graphics_data/barplot_data.csv'))
                x = data['iter'][1:]
                y = data[sel][1:]

            #plt.plot(x, y, marker='o', linestyle='dashed', label=dataset)
            plt.bar(x+width*mul, y*100, ec='k', alpha=0.8, hatch='//', width=width, label=dataset)
            mul+=1

        plt.legend()
        plt.ylim([0, 105])

        if linguagem == 'pt':
            plt.ylabel('Objetos selecionados da classe nova (%)', fontsize=15)
            plt.xlabel('Iteração', fontsize=15)
            plt.title('Métrica: ' + sel, fontsize=15)

        elif linguagem == 'en':
            plt.ylabel("New class' objects selected (%)", fontsize=15)
            plt.xlabel('Iteration', fontsize=15)
            plt.title('Metric: ' + sel, fontsize=15)
        plt.xticks(x+(width/2)*(len(datasets_list)-1),x)
        plt.savefig(results_dir+sel+'_overall_newclass_prop.png')

if __name__ == "__main__":

    #dataset_list = ['dp_ceratocystis1','dp_ceratocystis2','dp_ceratocystis5','dp_ceratocystis10','dp_ceratocystis20']
    #sel_list = ['entropia','silhueta0','silhueta1']
    #sel_list = ['entropia']
    #linguagem = 'pt'


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
    models_list = args.classifiers

    models_list = list(dict.fromkeys(models_list)) # remove palavras repetidas na lista
    results_dir = []

    for ml in models_list:

        script_dir = os.path.dirname(__file__)
        results_dir = os.path.join(script_dir, 'results/' + dataset_list[0].split('_')[0] + '_final_results/' + ml + '/')

        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        if ml == 'IC_EDS':
            sel = ['EDS']

        elif ml ==  'NNO' and linguagem == 'en':
            sel = ['unk_class_prob']
        elif ml ==  'NNO' and linguagem == 'pt':
            sel = ['prob_class_desc']
        else:
            if linguagem == 'en':
                if 'EDS' in sel_list:
                    sel = sel_list.copy()
                    sel.remove('EDS')
                if 'unk_class_prob' in sel_list:
                    sel = sel_list.copy()
                    sel.remove('unk_class_prob')
                else:
                    sel = sel_list.copy()
            elif linguagem == 'pt':
                if 'EDS' in sel_list:
                    sel = sel_list.copy()
                    sel.remove('EDS')
                if 'prob_class_desc' in sel_list:
                    sel = sel_list.copy()
                    sel.remove('prob_class_desc')
                else:
                    sel = sel_list.copy()



        plot_overall_datasets_accuracy(dataset_list,sel,linguagem, results_dir, ml)
        plot_overall_datasets_new_class_error(dataset_list,sel,linguagem, results_dir, ml)
        plot_overall_datasets_new_class_prop(dataset_list,sel,linguagem, results_dir, ml)

