# ------------------------------------------------------------------------------
# USEFUL FUNCTIONS
# > These functions are required in different places in the code
# ------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold, KFold
from itertools import combinations
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
from scipy.spatial import distance
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances
import torchvision.transforms as transforms
from ST_modules.data_loader import RODFolder


class ST_functions:
    # ------------------------------------------------------------------------------
    # MAIN FUNCTIONS
    # ------------------------------------------------------------------------------

    def __init__(self):
        pass

    '''
    def separate_features_and_labels(self,train_path, test_path, class_index, class2drop = None, scale = False):

        class_index = class_index
        df_training = pd.read_csv(train_path)
        scaler = MinMaxScaler()


        if class2drop != None:
            df_valores = df_training.loc[df_training['class'] == class2drop]
            df_training.drop(df_valores.index, inplace=True)

        feat_index = list(range(df_training.shape[1]))
        feat_index.remove(class_index)
        train = df_training.iloc[:, feat_index].values
        train_labels = df_training.iloc[:, class_index].values
        if scale == True:
            train_scaler = scaler.fit(train)
            train = train_scaler.transform(train)

        df_test = pd.read_csv(test_path)
        feat_index = list(range(df_test.shape[1]))
        feat_index.remove(class_index)
        test = df_test.iloc[:, feat_index].values
        test_labels = df_test.iloc[:, class_index].values
        if scale == True:
            test_scaler = scaler.fit(test)
            test = test_scaler.transform(test)

        return train, train_labels, test, test_labels

    def train_and_test_set_generator(self, dataset_path, dataset_name, class2drop, num_features, prop_train_data):

        data = pd.read_csv(dataset_path, header=None)

        target_values = data.iloc[:, num_features].values
        target_values = np.unique(target_values)

        train_classes = []
        test_classes = []

        for i in target_values:

            pd_class = data.loc[data[num_features] == i]
            n_obj_class = int(pd_class.shape[0] * prop_train_data)
            sort_class = random.sample(range(pd_class.index[0], pd_class.index[-1] + 1), n_obj_class)

            train_classes.append(pd_class.loc[sort_class].copy())
            test_classes.append(pd_class.drop(sort_class).copy())

        train_set = pd.concat(train_classes, axis=0)
        test_set = pd.concat(test_classes, axis=0)

        df_val = train_set.loc[train_set[num_features] == class2drop]
        train_set.drop(df_val.index, inplace=True)

        train_set.to_csv('data/'+dataset_name+'_train.csv', index = False)
        test_set.to_csv('data/'+dataset_name+'_test.csv', index=False)
    '''

    #Funções para uso do NNO com rede neural
    #-------------------------------------------------------------------
    def make_dataset(self):

        transform_train = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomCrop(64, padding=8, padding_mode='edge'),

        ])

        transform_train = transforms.Compose([transform_train,
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5), (1., 1., 1.)),
                                              ])

        transform_basic = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomCrop(64, padding=8, padding_mode='edge'),
            transforms.RandomHorizontalFlip(),
        ])

        transform_val = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomCrop(64, padding=8, padding_mode='edge'), ])

        transform_val = transforms.Compose([transform_val,
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (1., 1., 1.)),
                                            ])

        transform_test = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1., 1., 1.)),
        ])

        start = 0
        end = 1000

        return transform_train, transform_basic, transform_val, transform_test,start, end

    #Funções para uso do classificador incremental
    #--------------------------------------------------------------
    def get_pseudopoints(self, train):

        kmeans = KMeans(n_clusters=10,  # numero de clusters
                        init='k-means++', n_init=10,
                        # método de inicialização dos centróides que permite convergencia mais rápida
                        max_iter=300)  # numero de iterações do algoritmo

        # Visualização do K-means para os dois conjuntos de dados

        pred_train = kmeans.fit_predict(train)
        kmeans_train_center = kmeans.cluster_centers_
        # objs_train_to_center_clusters = kmeans.fit_transform(train) #calcula a distancia de cada ponto até os centros de cada cluster

        n_centros = len(kmeans_train_center)

        from collections import Counter, defaultdict
        # print(Counter(pred_train))
        # print(Counter(pred_train)[1])

        train_center_dists = []
        for k in train:
            col = np.zeros(shape=(len(kmeans_train_center)))

            for c in range(len(kmeans_train_center)):
                d = np.linalg.norm(k - kmeans_train_center[c])
                col[c] = d

            train_center_dists.append(col)

        def find_rad(dist_matrix, obj_per_cluster):
            rad = []
            n_clusters = np.shape(dist_matrix)[1]

            for i in range(n_clusters):
                dist_matrix[:, i].sort()
                dist = dist_matrix[:, i]
                dist_ci = dist[:obj_per_cluster[i]]
                rad.append(max(dist_ci))

            return rad

        rad = find_rad(np.array(train_center_dists), Counter(pred_train))

        pseudopoints = []

        for i in range(n_centros):
            a = [kmeans_train_center[i], rad[i], Counter(pred_train)[i]]
            pseudopoints.append(a.copy())



        return pseudopoints, n_centros, pred_train, kmeans_train_center, train_center_dists

    def get_mean_class(self,train, train_labels):

        mean = []
        old_class = []
        labels = np.unique(train_labels)
        # print(train, train.shape, type(train))

        for l in labels:
            ind = np.where(train_labels == l)[0]
            # print(ind)
            objs = train[ind]
            # print('sh', np.shape(objs))

            old_class.append(l)
            mean.append(np.mean(objs, axis=0))  # media original sobre todas as amostras da classe

        # old_class = train_labels


        # print(mean)

        return mean, old_class

    def upg_means(self,old_class, mean, train, train_labels, nb_exemplars):

        labels = np.unique(train_labels)

        for l in labels:

            ind = np.where(train_labels == l)[0]
            objs = train[ind]
            # print(objs)

            if not (l in old_class):
                old_class.append(l)

                mean.append(np.mean(objs, axis=0))


            elif len(ind) < nb_exemplars:
                id = np.where(old_class == l)[0][0]
                # print(id, old_class, mean)
                mean[id] = np.mean(objs, axis=0)
                print('new', mean[id])

        # print(mean)

        return old_class, mean

    def upg_pseudopoints(self,old_class, pseudopoints, train, train_labels):

        labels = np.unique(train_labels)

        for l in labels:
            # print(l in old_class)
            if not (l in old_class):

                ind = np.where(train_labels == l)[0]

                mean = np.mean(train[ind], axis=0)
                # print('mean ', mean)
                nb_obj = len(ind)
                dists = [np.sqrt(np.sum((x - mean) ** 2, axis=0)) for x in train[ind]]
                rad = max(dists)
                a = [mean, rad, nb_obj]

                pseudopoints.append(a.copy())


            else:

                ind = np.where(train_labels == l)[0]

                objs = train[ind]

                for o in objs:

                    belong = False
                    for ps in pseudopoints:
                        # print(o, ps[0])
                        # print(np.sqrt(np.sum((o - ps[0])**2,axis=0)))

                        if np.sqrt(np.sum((o - ps[0]) ** 2, axis=0)) <= ps[1]:
                            belong = True

                    if belong == False:
                        # print('extend')
                        vec_dist = [np.sqrt(np.sum((o - ps[0]) ** 2, axis=0)) for ps in pseudopoints]
                        pseudopoints[np.argmin(vec_dist)][1] = min(vec_dist)


        # print(pseudopoints, np.shape(pseudopoints))
        return pseudopoints

    # teste incremental
    def kms_for_new_class(self, pseudopoints, test, kmeans_approach):

        kmeans_test = KMeans(n_clusters=12,  # numero de clusters
                             init='k-means++', n_init=10,
                             # método de inicialização dos centróides que permite convergencia mais rápida
                             max_iter=300)  # numero de iterações do algoritmo

        # Visualização do K-means para os dois conjuntos de dados

        pred_test = kmeans_test.fit_predict(test)
        kmeans_test_center = kmeans_test.cluster_centers_
        objs_test_to_center_clusters = kmeans_test.fit_transform(test)

        data_centers = np.array([centroids[0] for centroids in pseudopoints])
        rad_clusters = np.array([ps[1] for ps in pseudopoints])

        # print(data_centers, np.shape(data_centers))
        new_data_labels, new_centers_kept, silhouette, nearest_data_object_position = self.fd_novelties([], [],
                                                                                                          data_centers,
                                                                                                          [], test,
                                                                                                          pred_test,
                                                                                                          kmeans_test_center,
                                                                                                          objs_test_to_center_clusters,
                                                                                                        rad_clusters,
                                                                                                          kmeans_approach,
                                                                                                          0.8)

        return silhouette

    def fd_novelties(self, data, data_labels, data_centers, data_dists, new_data, new_data_labels,
                       new_data_centers,
                       new_data_dists, rad_clusters, kmeans_approach, thrs):

        # the higher the value the greater the qualification to be a new cluster object
        threshold = thrs

        # 0: new obj center X nearest data obj
        # 1: new obj center X nearest data center
        # 2: new obj center * new objs density X nearest data center * data density
        # 3: new obj center * (std new obj cluster / std near data cluster) X near data center * (std near data cluster / std new obj cluster)
        approach = kmeans_approach

        silhouette = [-1] * len(new_data_labels)
        new_centers_kept = [-1] * len(new_data_centers)

        nearest_data_object_position = list()
        min_val = 0.0000000001
        # print(new_data_labels)

        for i in range(len(new_data_labels)):
            #    if i%50 ==0: print(i)

            ### new object distance to the center of its cluster
            # dist_new_obj_center = new_data_dists[i, new_data_labels[i]]

            dist_new_obj_center = np.min(new_data_dists[i])
            #     print([i,new_data_labels[i]])


            ### new object distance to the nearest center calculated from existing data
            dist_data_centers = []
            for j in range(len(data_centers)):
                dist_data_centers.append(np.linalg.norm(new_data[i, :] - data_centers[j, :]))
            nearest_data_cluster = np.argmin(dist_data_centers)  # label nearest data cluster
            dist_nearest_data_center = dist_data_centers[nearest_data_cluster]

            ### find distance of the new object to the nearest object belonging to the nearest data cluster

            distance_nearest_data_object = dist_nearest_data_center - rad_clusters[nearest_data_cluster]

            a0 = dist_new_obj_center
            b0 = distance_nearest_data_object

            a1 = dist_new_obj_center
            b1 = dist_nearest_data_center

            a2 = 0  # dist_new_obj_center * density_new_objs
            b2 = 0  # dist_nearest_data_center * density_data

            a3 = 0  # dist_new_obj_center * (std_new_objs / (std_data + min_val))
            b3 = 0  # dist_nearest_data_center * (std_data / (std_new_objs + min_val))

            sil_calc_terms = np.array([[a0, b0], [a1, b1], [a2, b2], [a3, b3]])

            # print(sil_calc_terms[approach, :])
            silhouette[i] = (sil_calc_terms[approach, 1] - sil_calc_terms[approach, 0]) / np.max(
                sil_calc_terms[approach, :])

            if silhouette[i] < threshold:
                new_data_labels[i] = nearest_data_cluster
            else:
                # when a new cluster is found (by keeping a new center)
                # a new label higher than all already annotated is defined for its objects
                # position p with -1 in new_centers_kept means that the p cluster in new data won't be kept
                if new_centers_kept[new_data_labels[i]] < 0:
                    new_centers_kept[new_data_labels[i]] = new_data_labels[i] + int(np.max(new_data_labels)) + 1
                new_data_labels[i] = new_centers_kept[new_data_labels[i]]
            # np.savetxt('silhouette.csv', silhouette, delimiter=',', fmt='%1.4f')
            # np.savetxt('silhouette_density.csv', silhouette_density, delimiter=',', fmt='%1.4f')
        # print(sum(silhouette)/len(silhouette))

        return new_data_labels, new_centers_kept, silhouette, nearest_data_object_position
    #-------------------------------------------------------------

    def select_exemplars(self, train, train_labels, mean, nb_exemplars, old_class):

        exemplars_set = []
        labels_set = []
        labels = np.unique(train_labels)
        # print('labels', labels)
        cm = 0
        # print(train, train.shape, type(train))

        for l in labels:
            ind = np.where(train_labels == l)[0]
            # print(ind)
            objs = train[ind]
            # print('sh', np.shape(objs))

            # print('teste', np.where(old_class == l)[0][0] )
            mean_aux = mean[np.where(old_class == l)[0][0]]  # pode dar erro aqui, pois as labels podem estar ordenadas de forma diferente dos elementos do vetor mean
            exemplars = []
            ex_lb = []
            # print('mean', mean)
            if np.shape(objs)[0] < nb_exemplars:

                [exemplars.append(x.copy()) for x in objs]  # New object to avoid passing by inference

                [ex_lb.append(l) for i in range(np.shape(objs)[0])]

            else:
                for k in range(1, nb_exemplars + 1):
                    # print(exemplars, np.shape(exemplars))
                    S = np.sum(exemplars, axis=0)  # [feature_dim] sum of selected exemplars vectors
                    mu_p = (objs + S) / k  # [n, feature_dim] sum to all vectors
                    # print(mu_p, np.shape(mu_p))
                    # print(mean, np.shape(mean))
                    # print('dists')
                    # print(np.sqrt(np.sum((mean_aux - mu_p) ** 2, axis=1)))
                    i = np.argmin(np.sqrt(np.sum((mean_aux - mu_p) ** 2, axis=1)))

                    exemplars.append(np.array(objs[i]))  # New object to avoid passing by inference
                    ex_lb.append(l)

                    objs = np.delete(objs, i, axis=0)  # Remove it to avoid duplicative selection

            # cm+=1
            exemplars_set.append(exemplars.copy())
            labels_set.append(ex_lb.copy())

        exemplars_set = np.concatenate(exemplars_set, axis=0)
        labels_set = np.concatenate(labels_set, axis=0)
        # print(exemplars_set, np.shape(exemplars_set))
        # print(labels_set, np.shape(labels_set))
        return exemplars_set, labels_set

    def nearest_mean_examplars(self, exemplars_set, ex_lb, test):

        class_means = []
        labels = np.unique(ex_lb)

        for l in labels:
            ind = np.where(ex_lb == l)[0]
            ex = exemplars_set[ind]
            class_means.append(np.mean(ex, axis=0))  # obs: pode dar erro aqui (fazer ex.copy())
        # print([np.sqrt(np.sum((x - class_means)**2,axis=1)) for x in test])

        probs = [self.class_prob_NCM(np.sqrt(np.sum((x - class_means) ** 2, axis=1))) for x in test]
        y = [labels[np.argmin(np.sqrt(np.sum((x - class_means) ** 2, axis=1)))] for x in test]

        return y,np.array(probs)

    def class_prob_NCM (self, dw_dists):
        # p(c|x) = exp(-dw_dists)/ sum(exp(-dw_dists))
        #https://patents.google.com/patent/US20140029839A1/en

        prob = np.exp(-1*dw_dists)/sum(np.exp(-1*dw_dists))

        return prob



    def sel_exemplares(self, train, train_labels, len_exemplares):

        #seleção por proximidade à média da classe
        #-----------------------------------------

        classes, count = np.unique(train_labels, return_counts=True)
        n = round(len_exemplares/len(classes)) # calcula numero de amostras por classe
        new_train = []
        new_train_labels = []

        for c in classes:

            if count[np.where(classes == c)] >= n:

                idx = np.array(np.where(train_labels==c))[0] # pega todos os indices dos objetos da classe c

                c_mean = np.mean(train[idx,:], axis=0) #calcula media da classe c


                dists_to_mean = [np.linalg.norm(train[x,:]-c_mean) for x in idx] #calcula distancia dos objetos à media

                ordered_obj = list(enumerate(dists_to_mean)) # organiza as distancias calculadas de forma crescente
                ordered_obj.sort(key=lambda x: x[1])

                argmins = [ob[0] for ob in ordered_obj] #extrai apenas os indices dos objetos ordenados

                objs = argmins[:n]


                new_train.extend(train[idx[objs],:])
                new_train_labels.extend(train_labels[idx[objs]])


            else:
                idx = np.array(np.where(train_labels==c))[0]  # pega todos os indices dos objetos da classe c
                new_train.extend(train[idx, :])


                new_train_labels.extend(train_labels[idx])



        # seleção aleatória para rehearsal
        #-------
        '''
        classes, count = np.unique(train_labels, return_counts=True)
        classes = classes.tolist()
        buff_objs = []
        buff_labels = []
        #len_exemplares = 0.2*np.shape(train)[0]
        n = round(len_exemplares / len(classes))
        # print(classes)
        # print(count)
        for i in range(len(classes)):
            # print(np.shape(train))
            if count[i] < n:

                aux = np.where(train_labels == classes[i])[0][:]

                # print('aux', aux[0][:], np.shape(aux[0][:]))
                buff_objs.extend(train[aux, :])

                buff_labels.extend([x[0] for x in train_labels[aux]])

                # print('boj: ', np.shape(train_labels[ax]), np.shape(train))
                # print(train_labels[aux], type(buff_labels))
                # print([list(x) for x in buff_objs])

                train = np.delete(train, aux, axis=0)
                train_labels = np.delete(train_labels, aux)

        # print(np.shape(train))

        for i in np.unique(buff_labels):
            classes.remove(i)

        # print(train_labels)
        # print(buff_labels)

        n = round(len_exemplares / len(classes))
        # print((classes))
        new_train = []
        new_train_labels = []

        for tl in classes:
            ind = np.where(train_labels == tl)
            rd_objs = np.random.choice(ind[0], n)
            new_train.extend(train[rd_objs][:])
            new_train_labels.extend(train_labels[rd_objs])

        new_train.extend(buff_objs)
        new_train_labels.extend(buff_labels)
        # print(new_train_labels)
        # print(np.shape(new_train), type(new_train))
        # print(new_train)

        '''
        #------------------------
        #print(np.array(new_train), np.array(new_train_labels))
        return np.array(new_train), np.array(new_train_labels)

    # Ordena os dados de teste de acordo com a classe
    def sort_testset(self, x_test, y_test):
        list_labels_test = y_test[:]
        ind = []
        ordered_y_test = []

        list_labels_test = list(enumerate(list_labels_test))
        list_labels_test.sort(key=lambda x: x[1])

        for x in list_labels_test:
            ind.append(x[0])
            ordered_y_test.append(x[1])

        ordered_x_test = x_test[ind]

        return ordered_x_test, np.array(ordered_y_test)

    def draw_new_classes(self, list_new_class_labels, ordered_x_test, ordered_y_test):

        new_classes_objs = []
        for nc in list_new_class_labels:
            # print(nc, ordered_y_test)
            ind = np.where(ordered_y_test == nc[1])
            if len(np.shape(ordered_x_test)) == 1:
                nc_objs = ordered_x_test[ind]
            else:
                nc_objs = ordered_x_test[ind, :]
            nc_labels = ordered_y_test[ind]

            ordered_x_test = np.delete(ordered_x_test, ind, axis=0)
            ordered_y_test = np.delete(ordered_y_test, ind, axis=0)

            new_classes_objs.append([nc_objs.copy(), nc_labels.copy()])

        return new_classes_objs, ordered_x_test, ordered_y_test

    def class_error(self, pred, test_labels, classe):
        c = 0
        dflabels = pd.DataFrame(test_labels)
        dfpred = pd.DataFrame(pred)

        ind = dflabels.loc[dflabels[0] == classe].index
        labels_val = dflabels.loc[dflabels[0] == classe].to_numpy()

        pred_val = dfpred.iloc[ind].to_numpy()

        if len(labels_val) == 0 or len(labels_val) == []:
            error = 1
        else:

            for i in range(len(labels_val)):
                if pred_val[i] == labels_val[i]:
                    c += 1

            error = 1 - (c / len(labels_val))

        return round(error, 3)

## ver depois como adequar essa função
#---------------------------------------
    '''
    def save_file(name_file, data, extension):
        if extension == 'txt':
            np.savetxt(name_file, data, delimiter='', fmt='%.4f')
            !cp  "$name_file" "/content/drive/MyDrive/MyFiles/PROJECTS/2021-Weed-Detection/Codes/data"
        else:
            if extension == 'csv':
                np.savetxt(name_file, data, delimiter=',', fmt='%.4f')
                !cp "$name_file" "/content/drive/MyDrive/MyFiles/PROJECTS/2021-Weed-Detection/Codes/data"
            else:
                if extension == 'png':
                    save_map(data, name_file)
                    !cp
                    "$name_file" "/content/drive/MyDrive/MyFiles/PROJECTS/2021-Weed-Detection/Codes/data"

    '''

    def get_batch_data(self, train_data_path, test_data_path, class_index, join_data, size_batch, iter, class2drop=-1, scale = False):

        scaler = MinMaxScaler()
        df_training = []
        train = []
        train_labels = []
        if type(train_data_path) != str:
            train = train_data_path[:,0]
            train_labels = train_data_path[:,1].astype(int) #converte todos os dados para tipo int
            #print(train,train_labels)

        elif train_data_path:
            #https://www.pythonanywhere.com/forums/topic/30323/
            #https://stackoverflow.com/questions/2083987/how-to-retry-after-exception
            #https://www.hashtagtreinamentos.com/try-e-except-no-python?gclid=Cj0KCQiAorKfBhC0ARIsAHDzslv8twjYhj3cBwnujIVtL_3wI15CAgFh8VukglyiajnKOO8y9ZNLu38aAhBzEALw_wcB

            for x in range(20): # 20 tentativas
                try:
                    #faça algo
                    df_training = pd.read_csv(train_data_path)  # , header=None)

                except OSError as error:
                    #se ocorrer o erro especificado faça
                    print(error)
                except:
                    #se ocorrer outro tipo de erro faça
                    print('Deu algum outro problema no pd.read_csv(train_data_path)')
                else:
                    #o try foi executado sem erros, então encerra o for
                    break



            feat_index = list(range(df_training.shape[1]))
            feat_index.remove(class_index)
            train = df_training.iloc[:, feat_index].values
            if scale == True:
                train_scaler = scaler.fit(train)
                train = train_scaler.transform(train)

            train_labels = df_training.iloc[:, class_index].values


        df_test = []
        test = []
        test_labels = []
        if test_data_path:
            #df_test = pd.read_csv(test_data_path)#, header=None)
            for x in range(20):
                try:
                    # faça algo
                    df_test = pd.read_csv(test_data_path)#, header=None)

                except OSError as error:
                    # se ocorrer o erro especificado faça
                    print(error)
                except:
                    # se ocorrer outro tipo de erro faça
                    print('Deu algum outro problema no pd.read_csv(test_data_path)')
                else:
                    # o try foi executado sem erros, então encerra o for
                    break

            # print(df_test.shape)
            feat_index = list(range(df_test.shape[1]))
            feat_index.remove(class_index)
            test = df_test.iloc[:, feat_index].values
            if scale == True:
                test_scaler = scaler.fit(test)
                test = test_scaler.transform(test)
            test_labels = df_test.iloc[:, class_index].values

        if join_data:
            data = np.concatenate([train, test])
            data_labels = np.concatenate([train_labels, test_labels])
        else:
            data = train
            data_labels = train_labels

        num_objects = data.shape[0]


        folds = round(num_objects / size_batch)



        '''skf = StratifiedKFold(n_splits=folds)
        for train, test in skf.split(data, data_labels):
        print(train)
        print(test)
        #print('train -  {}   |   test -  {}'.format(np.bincount(y[train]), np.bincount(y[test])))'''

        i = 1
        test_data_fold = []
        test_labels_fold = []
        train_data_fold = []
        train_labels_fold = []
        # X, y = data, data_labels
        skf = StratifiedKFold(n_splits=folds, random_state=None, shuffle=False)


        for test_index, train_index in skf.split(data, data_labels):
            # print(len(train_index))
            # print(len(test_index))

            if (iter == i):
                # print ("\nIteração = ", i)
                # print("TEST-DATA:", data[test_index], "\nTEST-LABELS:", data_labels[test_index])
                # print("TRAIN-DATA:", data[train_index], "\nTRAIN-LABELS:", data_labels[train_index])
                #print('train: ', len(np.unique(train_index)))
                #print(train_index)
                #print('test: ', len(np.unique(test_index)))
                #print(test_index)
                train_data_fold = data[train_index]
                train_labels_fold = data_labels[train_index]
                test_data_fold = data[test_index]
                test_labels_fold = data_labels[test_index]

            i = i + 1

        # aqui tinha que ter uma forma de remover todos de uma certa classe no conjunto de treino
        # com opção de remover ou não (tipo um parâmetro = -1 não remove, mas se for um valor positivo remove essa classe)
        if class2drop != -1:
            indexes = np.where(train_labels_fold == class2drop)
            train_data_fold = np.delete(train_data_fold, indexes, axis=0)
            train_labels_fold = np.delete(train_labels_fold, indexes)

        #print('\ntrain objs: ', train_data_fold.shape, train_labels_fold)

        '''while (train_index, test_index in skf.split(X, y)) and (iter<i):
            i= i+1
            print ("\nIteração= ", i)
            print("TRAIN:", data[train_index], "\nTEST:", data_labels[test_index])
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            #print("X_Test:", X_test)
            #print("y_Test:", y_test)
            #print (X_test.shape[0])
            #print (y_test.shape[0])
            print (data[train_index].shape[0])
            break'''

        '''for train_index, test_index in skf.split(X, y):
            i= i+1
            print ("\nIteração= ", i)
            print("TRAIN:", data[train_index], "\nTEST:", data_labels[test_index])
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            #print("X_Test:", X_test)
            #print("y_Test:", y_test)
            #print (X_test.shape[0])
            #print (y_test.shape[0])
            print (data[train_index].shape[0])'''

        return train_data_fold, train_labels_fold, test_data_fold, test_labels_fold



    # def hyperparametersTuning(...):
    # TO DO
    # https://scikit-learn.org/stable/modules/grid_search.html
    # https://www.kaggle.com/udaysa/svm-with-scikit-learn-svm-with-parameter-tuning


    ### Generating subset of features randomly
    def features_subset(self, tot_features, num_subsetfeat):
        features_list = list(range(0, tot_features))
        comb = combinations(features_list, num_subsetfeat)
        # perm = permutations(features_list, num_subsetfeat) # order matters, e.g.: (0,1) <> (1,0)
        subsetfeat_list = []
        for i in list(comb):
            subsetfeat_list.append(i)
        # ----------------------------------------------------------------------------------------------
        ### Versão baseada em sortear permutação e realizar um cortes na proporcao de 'size_subsetfeat'
        # features_list = list(range(0, num_features))
        #### INEFICIENTE, LISTA COM TODAS AS PERMUTAÇÕES POSSÍVEIS DE N FEATURES!!!
        # feat_perm = [p for p in permutations(features_list)]
        # size_feat_perm = len(list(feat_perm))
        # num_feat_perm = randrange(size_feat_perm)
        # sel_feat_perm = list(feat_perm)[num_feat_perm]
        # print(sel_feat_perm)
        # subsetfeat_list = []
        # int_size_subsetfeat = floor(len(sel_feat_perm) * size_subsetfeat)
        # for n in range(int_size_subsetfeat, num_features + 1, int_size_subsetfeat):
        #    subsetfeat_list.append(sel_feat_perm[n - int_size_subsetfeat:n])
        #    # print(sel_feat_perm[n-int_size_subsetfeat:n])
        # ----------------------------------------------------------------------------------------------
        return subsetfeat_list


    def clusterEnsemble(self, data):
        ssfeat_list = self.features_subset(data.shape[1], 2)
        max_k = int(len(data) ** (1 / 3))  # equal to cubic root # int(math.sqrt(len(apat_iceds_norm)))
        num_init = 5  # 20
        range_n_clusters = list(range(2, max_k))

        silhouette_list = []
        clusterers_list = []
        cluslabels_list = []
        nuclusters_list = []

        matDist = np.array(euclidean_distances(data, data))

        for n_size_ssfeat in range(int(len(ssfeat_list))):

            # Subconjunto de features
            subset_feat = ssfeat_list[n_size_ssfeat]
            X = data[:, subset_feat]

            best_silhouette_avg = -1.0
            best_clusterer = []
            best_cluster_labels = []
            best_num_clusters = -1

            for n_clusters in range_n_clusters:
                for n_init in range(num_init):

                    # Initialize the clusterer with n_clusters value and a random generator
                    # seed of 10 for reproducibility.
                    clusterer = KMeans(n_clusters=n_clusters, init='random')
                    cluster_labels = clusterer.fit_predict(X)

                    # The silhouette_score gives the average value for all the samples.
                    # This gives a perspective into the density and separation of the formed clusters
                    silhouette_avg = silhouette_score(X, cluster_labels)

                    if (silhouette_avg > best_silhouette_avg):
                        best_silhouette_avg = silhouette_avg
                        best_clusterer = clusterer
                        best_cluster_labels = cluster_labels
                        best_num_clusters = n_clusters

                    # print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
                    # clusterer_plots(X, cluster_labels, n_clusters, clusterer)

            silhouette_list.append(best_silhouette_avg)
            clusterers_list.append(best_clusterer)
            cluslabels_list.append(best_cluster_labels)  ### vai usar para gera a matriz de similaridades abaixo
            nuclusters_list.append(best_num_clusters)

        ############# CONSENSO ###################
        cluslabels_list = np.array(cluslabels_list)
        caMatrix = np.array([[0] * cluslabels_list.shape[1]] * cluslabels_list.shape[1])

        for i in range(cluslabels_list.shape[
                           0]):  # for (int i = 0; i < cluEnsemble.length; i++) {  ### TAMANHO DA LISTA cluslabels_list
            for j in range(cluslabels_list.shape[
                               1]):  # for (int j = 0; j < data.numInstances(); j++) { ### len(cluslabels_list[0])
                for k in range(cluslabels_list.shape[
                                   1]):  # for (int k = 0; k < data.numInstances(); k++) { ### len(cluslabels_list[0])
                    if cluslabels_list[i][j] == cluslabels_list[i][
                        k]:  ######## cluslabels_list[i][j] == cluslabels_list[i][k]
                        caMatrix[j][k] += 1
                    if i == cluslabels_list.shape[0] - 1:
                        caMatrix[j][k] = caMatrix[j][k] / cluslabels_list.shape[0]  ### TAMANHO DA LISTA cluslabels_list
        # print("Best Silhoutte =", silhouette_list, " Number of Clusters =", nuclusters_list)
        return [silhouette_list, clusterers_list, cluslabels_list, nuclusters_list, caMatrix, matDist]


    def remove_class(self, hidden_class, train, train_labels):
        train_labels.columns = ['Class']
        labeled_data = pd.concat([train, train_labels], axis=1, sort=False)
        labeled_data = labeled_data[labeled_data.Class != hidden_class]

        t = labeled_data.iloc[:, :-1]
        tl = labeled_data.iloc[:, -1:]

        tl.columns = [0]

        return [t, tl]


    def increment_training_set(self, sel_objects, train, train_labels, test, test_labels, iter, save_dir):
        if len(train[0]) <= 2:

            self.visualize_data(test, test_labels, sel_objects, iter, save_dir)

        test = pd.DataFrame(test)
        test_labels = pd.DataFrame(test_labels)
        objects = test.iloc[sel_objects, :]
        objects_labels = test_labels.iloc[sel_objects, :]

        str_model = save_dir.split('/') [-1]
        if str_model != 'NNO' and str_model != 'deepncm':
            # print("Selected Objects Classes: " + str(objects_labels.values.ravel()))
            train = pd.DataFrame(train)
            train_labels = pd.DataFrame(train_labels)
            train.columns = objects.columns
            train_labels.columns = objects_labels.columns
            tr = pd.concat([train, objects], axis=0)
            trl = pd.concat([train_labels, objects_labels], axis=0)
            te = test.drop(test.index[sel_objects])
            tel = test_labels.drop(test_labels.index[sel_objects])
        else:

            tr = objects
            trl = objects_labels
            te = test.drop(test.index[sel_objects])

            tel = test_labels.drop(test_labels.index[sel_objects])



        return [tr.to_numpy(), trl.to_numpy(), te.to_numpy(), tel.to_numpy(), objects_labels.to_numpy()]

    def class_proportion_objects(self, objs_labels, labels):

        classes = np.unique(labels)

        props = []
        for u in classes:
            r = np.count_nonzero(objs_labels == u)/len(objs_labels)
            props.append(r)

        return props




    def reduce_matrix(self, sel_objects, SSet):
        sim_mat = np.delete(SSet, np.s_[sel_objects], axis=0)
        sim_mat = np.delete(sim_mat, np.s_[sel_objects], axis=1)
        return sim_mat



    def calc_density(self, s):
        h = 5
        d = [0] * s.shape[0]
        for i in range(s.shape[0]):
            d[i] = np.sum(s[i, :][s[i, :].argsort()[h * (-1):]]) / h
        return d


    def calc_low_density(self, d):
        h = 5
        l = [0] * d.shape[0]
        for i in range(d.shape[0]):
            l[i] = np.sum(d[i, :][d[i, :].argsort()[h * (-1):]]) / h
        return l


    def c3e_sl(self, piSet, SSet, I, alpha):
        N = len(piSet)
        c = len(piSet[0, :])
        # piSet = np.array(piSet)
        y = [[1] * c] * N
        y = np.divide(y, c)
        labels = [-1] * N
        # y = pd.DataFrame(y)
        for k in range(0, I):
            for j in range(0, N):

                diffi = np.arange(0, N)
                cond = diffi != j
                t1 = np.array(SSet[j][cond])
                # http://mathesaurus.sourceforge.net/matlab-numpy.html
                # https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
                p1 = (np.transpose(t1 * np.ones([c, 1])) * y[cond, :]).sum(axis=0)
                p2 = sum(t1)
                y[j, :] = (piSet[j, :] + (2 * alpha * p1)) / (1 + 2 * alpha * p2)
                #labels[j] = int(np.where(y[j, :] == np.max(y[j, :]))[0])
                labels[j] = np.argmax(y[j, :])+1 #As classes do problema são 1,2 e 3
        return y, labels


    def eds(self, e, d, p, SSet):
        ### entropy measuse
        w = []
        for i in range(1, p + 1):

            ed = np.multiply(e, d)

            if i == 1:
                res = ed
                w.append(np.argmax(res))
            else:
                S_ir = SSet[:, w]
                S_ir = np.sum(S_ir, axis=1)
                S_mean = np.divide(S_ir, (i - 1))
                res = np.multiply(ed, (1 - S_mean))
                w.append(np.argmax(res))

            e[np.argmax(res)] = -1.E12

        return w


    def ic(self, probs, SSet, train, train_labels, test, test_labels):
        y = self.c3e_sl(probs, SSet, 5, 0.001)
        for k in range(10):
            e = self.calc_class_entropy(y)
            d = self.calc_density(SSet)
            w = self.eds(e, d, 5, SSet)
            [train, train_labels, test, test_labels] = self.increment_training_set(w, train, train_labels, test, test_labels)
            probs = self.svmClassification(train, train_labels, test, test_labels)
            SSet = self.reduce_matrix(w, SSet)
            y = self.c3e_sl(probs, SSet, 5, 0.001)
            print("Iteration " + str(k + 1) + " - Sizes: Training Set " + str(len(train)) + " - Test Set " + str(len(test)))


    ### ====================================================================================================================
    ### Functions for data visualization
    ### --------------------------------------------------------------------------------------------------------------------


    def visualize_data(self, X, labels, med_ind_list, k, save_dir):
        '''
        color_discrete_map = {'-3': 'rgb(255,255,0)',
                              'Centers': 'rgb(0,0,0)',
                              '-1': 'rgb(255,0,0)',
                              'Cluster 0': 'rgb(0,138,0)',
                              'Cluster 1': 'rgb(168, 2, 2)',
                              'Cluster 2': 'rgb(0,0,133)',
                              'Cluster 3': 'rgb(189,91,0)',
                              'Cluster 4': 'rgb(130,0,156)'}

        '''

        #Ploty - legend -> https://plotly.com/python/legend/
        #PLoty - discrete colors -> https://plotly.com/python/discrete-color/
        color_discrete_map = {'0.0': '#1F77B4',
                              '1.0': '#FF7F0E',
                              '2.0': '#2CA02C',
                              '3.0': '#D62728',
                              '4.0': '#9467BD',
                              '5.0': '#8C564B',
                              '6.0': '#E377C2',
                              '7.0': '#7F7F7F',
                              '8.0': '#BCBD22',
                              '9.0': '#17BECF',
                              'Selected': 'black'}

        if type(labels[0]) != np.ndarray:
            df_X = pd.DataFrame({'X': X[:, 0], 'Y': X[:, 1], 'Labels': [str(labels[i]) for i in range(len(labels))]})

        else:
            df_X = pd.DataFrame({'X': X[:, 0], 'Y': X[:, 1], 'Labels': [str(labels[i][0]) for i in range(len(labels))]})


        if (med_ind_list != []):
            med_labels = np.array([-1] * len(med_ind_list))
            med_labels = ['Selected' for i in med_labels]
            df_M = pd.DataFrame({'X': X[med_ind_list, 0], 'Y': X[med_ind_list, 1],
                                 'Labels': med_labels})#'Labels': med_labels.astype(str)})
            df_X = pd.concat([df_X, df_M])

        fig = px.scatter(df_X, x='X', y='Y', color='Labels',
                         color_discrete_map=color_discrete_map, width=1000, height=800)
        fig.update_traces(marker=dict(size=8), line=dict(color='rgb(0,0,0)', width=4),
                          selector=dict(mode='Masked'))
        fig.update_layout(
            legend=dict(
            font=dict(
            family="Courier",
            size=18,
            color="black"
        )))
        ### TENTE AI INVÉS DO FIG.SHOW() USAR UMA LINHA PARA SALVAR EM DISCO EM ALGUMA EXTENSÃO DE IMAGEM.
        #fig.show()
        fig.write_image(save_dir+"/selection_"+str(k)+".png")
        # fig.write_image("selection_"+str(k)+".png")
        # drive.mount('drive')
        # images_dir = '/content/drive/MyDrive/MyFiles/PROJECTS/2020-Self-Training/Codes/fce-self-training-graficos'
        # plt.savefig(f"{images_dir}/abc.png")

        # plt.savefig('name.png')
        # files.download('name.png')

    def find_novelties(self, data, data_labels, data_centers, data_dists, new_data, new_data_labels, new_data_centers,
                       new_data_dists, kmeans_approach, thrs):

        # the higher the value the greater the qualification to be a new cluster object
        threshold = thrs

        # 0: new obj center X nearest data obj
        # 1: new obj center X nearest data center
        # 2: new obj center * new objs density X nearest data center * data density
        # 3: new obj center * (std new obj cluster / std near data cluster) X near data center * (std near data cluster / std new obj cluster)
        approach = kmeans_approach

        silhouette = [-1] * len(new_data_labels)
        new_centers_kept = [-1] * len(new_data_centers)

        nearest_data_object_position = list()
        min_val = 0.0000000001
        # print(new_data_labels)

        for i in range(len(new_data_labels)):
            #    if i%50 ==0: print(i)

            ### new object distance to the center of its cluster
            # dist_new_obj_center = new_data_dists[i, new_data_labels[i]]

            dist_new_obj_center = np.min(new_data_dists[i])
            #     print([i,new_data_labels[i]])

            ### new object distance to the nearest center calculated from existing data
            dist_data_centers = []
            for j in range(len(data_centers)):
                dist_data_centers.append(np.linalg.norm(new_data[i, :] - data_centers[j, :]))
            nearest_data_cluster = np.argmin(dist_data_centers)  # label nearest data cluster
            dist_nearest_data_center = dist_data_centers[nearest_data_cluster]

            ### find distance of the new object to the nearest object belonging to the nearest data cluster
            objects_nearest_data_cluster = data[data_labels == nearest_data_cluster, :]
            dists_obs_nearest_data_cluster = distance.cdist(objects_nearest_data_cluster, np.array([new_data[i, :]]),
                                                            'euclidean')
            #print(dists_obs_nearest_data_cluster)
            nearest_data_object = np.argmin(dists_obs_nearest_data_cluster)
            nearest_data_object_position.append(objects_nearest_data_cluster[nearest_data_object])
            distance_nearest_data_object = dists_obs_nearest_data_cluster[nearest_data_object][0]

            # obtaining weights which can balance silhouette metric
            num_new_objs = sum(list(new_data_labels == new_data_labels[i]))
            num_data_objs = sum(list(data_labels == nearest_data_cluster))

            # dists_new_objs = np.sum(new_data_dists[list(new_data_labels == new_data_labels[i]), new_data_labels[i]])
            dists_new_objs = np.sum(
                new_data_dists[list(new_data_labels == new_data_labels[i]), np.argmin(new_data_dists[i])])

            for ob in range(len(data_labels)):
                if data_labels[ob] == nearest_data_cluster:
                    index = ob
                    break

            # dists_data = np.sum(data_dists[list(data_labels == nearest_data_cluster), nearest_data_cluster])
            dists_data = np.sum(data_dists[list(data_labels == nearest_data_cluster), np.argmin(data_dists[index])])

            density_new_objs = dists_new_objs / num_new_objs
            density_data = dists_data / num_data_objs

            # std_new_objs = np.std(new_data_dists[list(new_data_labels == new_data_labels[i]), new_data_labels[i]])
            std_new_objs = np.std(
                new_data_dists[list(new_data_labels == new_data_labels[i]), np.argmin(new_data_dists[i])])

            # std_data = np.std(data_dists[list(data_labels == nearest_data_cluster), nearest_data_cluster])
            std_data = np.std(data_dists[list(data_labels == nearest_data_cluster), np.argmin(data_dists[index])])

            a0 = dist_new_obj_center #* density_new_objs
            b0 = distance_nearest_data_object #* density_data

            a1 = dist_new_obj_center
            b1 = dist_nearest_data_center

            a2 = dist_new_obj_center * density_new_objs
            b2 = dist_nearest_data_center * density_data

            a3 = dist_new_obj_center * (std_new_objs / (std_data + min_val))
            b3 = dist_nearest_data_center * (std_data / (std_new_objs + min_val))

            sil_calc_terms = np.array([[a0, b0], [a1, b1], [a2, b2], [a3, b3]])

            # print(sil_calc_terms[approach, :])
            silhouette[i] = (sil_calc_terms[approach, 1] - sil_calc_terms[approach, 0]) / np.max(
                sil_calc_terms[approach, :])

            if silhouette[i] < threshold:
                new_data_labels[i] = nearest_data_cluster
            else:
                # when a new cluster is found (by keeping a new center)
                # a new label higher than all already annotated is defined for its objects
                # position p with -1 in new_centers_kept means that the p cluster in new data won't be kept
                if new_centers_kept[new_data_labels[i]] < 0:
                    new_centers_kept[new_data_labels[i]] = new_data_labels[i] + int(np.max(new_data_labels)) + 1
                new_data_labels[i] = new_centers_kept[new_data_labels[i]]
            # np.savetxt('silhouette.csv', silhouette, delimiter=',', fmt='%1.4f')
            # np.savetxt('silhouette_density.csv', silhouette_density, delimiter=',', fmt='%1.4f')
        # print(sum(silhouette)/len(silhouette))

        return new_data_labels, new_centers_kept, silhouette, nearest_data_object_position

        #
        # Main function incorporating new data into the database by which novelties can be detected

    def augment_data(self, num_ini_clu, data, data_labels, data_centers, data_dists, new_data, new_data_labels,
                     new_data_centers, new_data_dists, kmeans_approach, threshold):

        new_data_labels, new_centers_kept, silhouette, _ = self.find_novelties(data, data_labels, data_centers,
                                                                               data_dists,
                                                                               new_data,
                                                                               new_data_labels, new_data_centers,
                                                                               new_data_dists, kmeans_approach,
                                                                               threshold)

        # ------------------------------------------------------------------------------------------------------------------
        # Incorporating new data
        # ------------------------------------------------------------------------------------------------------------------
        data = np.concatenate([data, new_data])

        # ------------------------------------------------------------------------------------------------------------------
        # Arranging existing data centers with those new ones (from the new data) assigning new ordered labels to them
        # ------------------------------------------------------------------------------------------------------------------
        for i in range(len(new_centers_kept)):

            if new_centers_kept[i] >= 0:  # if higher than 0 means the new data cluster was kept (it is a new pattern!)
                data_centers = np.concatenate([data_centers, [new_data_centers[i, :]]])
                new_label = (data_centers.shape[
                                 0] - 1)  # finding its new label after joining it to existing data partition
                new_data_labels[new_data_labels == new_centers_kept[i]] = new_label

        # ------------------------------------------------------------------------------------------------------------------
        # Incorporating the new data labels to database labels structure
        # ------------------------------------------------------------------------------------------------------------------
        data_labels = np.concatenate([data_labels, new_data_labels])

        # ------------------------------------------------------------------------------------------------------------------
        # Recalculating centers, distances, and labels
        # ------------------------------------------------------------------------------------------------------------------
        data_centers, data_dists = self.recalc_centers(data_centers, data, data_labels)
        data_labels = self.recalc_labels(data_dists)

        # visualize_data(data, data_labels, [], data_centers, [])
        # visualize_graph(data_labels, num_ini_clu, data_centers)

        return data, data_labels, data_centers, data_dists, silhouette

    def recalc_centers(self, centers, data, labels):
        for j in range(len(centers)):
            centers[j, :] = data[labels == j].mean(0)
        dists = distance.cdist(data, centers, 'euclidean')
        return centers, dists

    def recalc_labels(self, dists):
        data_labels = []
        for i in range(dists.shape[0]):
            data_labels.append(np.argmin(dists[i]))
        return np.array(data_labels)

