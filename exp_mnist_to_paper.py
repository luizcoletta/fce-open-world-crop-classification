import os
import pandas as pd
import numpy as np

mnist = (['mnist2D_c0','mnist2D_c1','mnist2D_c2','mnist2D_c3','mnist2D_c4','mnist2D_c5',
          'mnist2D_c6','mnist2D_c7','mnist2D_c8','mnist2D_c9'])
selectors = ['entropy', 'silhouette', 'silh_mod']
metrics = ['Accuracy','Precision','F1-score','Recall']
mnist = (['mnist2D_c0','mnist2D_c1'])



path = os.path.dirname(__file__)
for x in metrics:
    res_entropia = []
    res_silh = []
    res_silh_mod = []
    for dataset in mnist:

        res_path = path+'/results/'+dataset+'/SVM/graphics_data/'+x+'_data.csv'

        acc_data = pd.read_csv(res_path)

        res_entropia.append(acc_data[selectors[0]].values) #eixo 0 é o experimento e o eixo 1 é a iteração
        #res_silh.append(acc_data[selectors[1]].values)
        #res_silh_mod.append(acc_data[selectors[2]].values)


    print(f'Resultados para {x} com {selectors[0]}:\n')
    means = [np.mean(values,axis=0) for values in zip(*res_entropia)] #faz a media dos valores de cada coluna
    ent_media_ini = round(means[0],4)
    ent_media_final = round(means[-1],4)
    ent_iter_res = [round(i[-1],4) for i in res_entropia]

    print(f'media das execuções na iteração 0: {ent_media_ini}')
    print(f'media das execuções na iteração final: {ent_media_final}')
    print(f'Valor final de cada execução: {ent_iter_res}\n\n')

    '''
    print(f'Resultados para {x} com {selectors[1]}:\n')
    means = [np.mean(values,axis=0) for values in zip(*res_silh)] #faz a media dos valores de cada coluna
    ent_media_ini = round(means[0],4)
    ent_media_final = round(means[-1],4)
    ent_iter_res = [round(i[-1],4) for i in res_silh]

    print(f'media das execuções na iteração 0: {ent_media_ini}')
    print(f'media das execuções na iteração final: {ent_media_final}')
    print(f'Valor final de cada execução: {ent_iter_res}\n\n')

    print(f'Resultados para {x} com {selectors[2]}:\n')
    means = [np.mean(values,axis=0) for values in zip(*res_silh_mod)] #faz a media dos valores de cada coluna
    ent_media_ini = round(means[0],4)
    ent_media_final = round(means[-1],4)
    ent_iter_res = [round(i[-1],4) for i in res_silh_mod]

    print(f'media das execuções na iteração 0: {ent_media_ini}')
    print(f'media das execuções na iteração final: {ent_media_final}')
    print(f'Valor final de cada execução: {ent_iter_res}\n\n')
    '''
    print('---------------------------------------------')











