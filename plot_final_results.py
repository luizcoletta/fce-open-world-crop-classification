import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def plot_overall_datasets_accuracy(datasets_list, sel_list, linguagem):

    # Gera gráfico com a acurácia de todos os datasets
    #Um gráfico para cada medida de seleção


    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'results/')

    for sel in sel_list:
        plt.figure(figsize=(10,6))
        for dataset in datasets_list:

            plt.rcParams['xtick.labelsize'] = 13
            plt.rcParams['ytick.labelsize'] = 13

            if linguagem == 'pt':
                data = pd.read_csv(os.path.join(results_dir,dataset+'/graphics_data/Acurácia_data.csv'))
                x = data['iter']
                y = data[sel]
            elif linguagem == 'en':
                data = pd.read_csv(os.path.join(results_dir, dataset + '/graphics_data/Accuracy_data.csv'))
                x = data['iter']
                y = data[sel]

            plt.plot(x, y, marker='o', linestyle='dashed', label=dataset)


        plt.legend()
        if linguagem == 'pt':
            plt.ylabel('Acurácia', fontsize=15)
            plt.xlabel('Iteração', fontsize=15)
            plt.title('Desempenho usando ' + sel, fontsize=15)

        elif linguagem == 'en':
            plt.ylabel('Accuracy', fontsize=15)
            plt.xlabel('Iteration', fontsize=15)
            plt.title('Performance with ' + sel, fontsize=15)

        plt.savefig(results_dir+sel+'_overall_accuracy.png')

def plot_overall_datasets_new_class_error(datasets_list, sel_list, linguagem):

    # Gera gráfico com a acurácia de todos os datasets
    #Um gráfico para cada medida de seleção


    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'results/')

    for sel in sel_list:
        plt.figure(figsize=(10,6))
        for dataset in datasets_list:

            plt.rcParams['xtick.labelsize'] = 13
            plt.rcParams['ytick.labelsize'] = 13

            if linguagem == 'pt':
                data = pd.read_csv(os.path.join(results_dir,dataset+'/graphics_data/error_class_3.csv'))
                x = data['iter']
                y = data[sel]
            elif linguagem == 'en':
                data = pd.read_csv(os.path.join(results_dir, dataset + '/graphics_data/error_class_3.csv'))
                x = data['iter']
                y = data[sel]

            plt.plot(x, y, marker='o', linestyle='dashed', label=dataset)


        plt.legend()
        if linguagem == 'pt':
            plt.ylabel('Erro da classe nova', fontsize=15)
            plt.xlabel('Iteração', fontsize=15)
            plt.title('Métrica: ' + sel, fontsize=15)

        elif linguagem == 'en':
            plt.ylabel("New class' error", fontsize=15)
            plt.xlabel('Iteration', fontsize=15)
            plt.title('Metric: ' + sel, fontsize=15)

        plt.savefig(results_dir+sel+'_overall_newclass_error.png')

def plot_overall_datasets_new_class_prop(datasets_list, sel_list, linguagem):

    # Gera gráfico com a acurácia de todos os datasets
    #Um gráfico para cada medida de seleção


    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'results/')

    for sel in sel_list:
        plt.figure(figsize=(10,6))
        mul = 0
        for dataset in datasets_list:

            width = 0.3
            plt.rcParams['xtick.labelsize'] = 13
            plt.rcParams['ytick.labelsize'] = 13

            if linguagem == 'pt':
                data = pd.read_csv(os.path.join(results_dir,dataset+'/graphics_data/barplot_data.csv'))
                x = data['iter']
                y = data[sel]
            elif linguagem == 'en':
                data = pd.read_csv(os.path.join(results_dir, dataset + '/graphics_data/barplot_data.csv'))
                x = data['iter']
                y = data[sel]

            #plt.plot(x, y, marker='o', linestyle='dashed', label=dataset)
            plt.bar(x+width*mul, y, ec='k', alpha=0.3, hatch='//', width=width, label=dataset)
            mul+=1

        plt.legend()
        if linguagem == 'pt':
            plt.ylabel('Proporção de objetos da classe nova', fontsize=15)
            plt.xlabel('Iteração', fontsize=15)
            plt.title('Métrica: ' + sel, fontsize=15)

        elif linguagem == 'en':
            plt.ylabel("New class' object proportion", fontsize=15)
            plt.xlabel('Iteration', fontsize=15)
            plt.title('Metric: ' + sel, fontsize=15)
        plt.xticks(x+(width/2)*(len(datasets_list)-1),x)
        plt.savefig(results_dir+sel+'_overall_newclass_prop.png')

if __name__ == "__main__":

    #dataset_list = ['dp_ceratocystis1','dp_ceratocystis2']
    #sel_list = ['entropia','silhueta0','silhueta1']
    #sel_list = ['entropia']
    #linguagem = 'pt'

    parser = argparse.ArgumentParser(description='Implementação de modelo Open-World para aprendizagem de novas ameaças na lavoura')


    parser.add_argument('-datasets', metavar='datasets', action='store', nargs='+', type=str, default=['dp_ceratocystis1'],
                        help='Classificadores usados no modelo')
    parser.add_argument('-selection', metavar='selection', action='store', nargs='+', type=str, default=['entropia'],
                        help='Método de seleção da nova classe')
    parser.add_argument('-language', metavar='language', action='store', type=str, default='pt',
                        help="Idioma dos resultados gerados ('pt' ou 'en')")

    args = parser.parse_args()

    dataset_list = args.datasets
    sel_list = args.selection
    linguagem = args.language

    plot_overall_datasets_accuracy(dataset_list,sel_list,linguagem)
    plot_overall_datasets_new_class_error(dataset_list,sel_list,linguagem)
    plot_overall_datasets_new_class_prop(dataset_list,sel_list,linguagem)

