# ------------------------------------------------------------------------------
# USEFUL FUNCTIONS
# > These functions are required in different places in the code
# ------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from itertools import combinations
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
from scipy.spatial import distance


'''

def file_exists(filePath):
    try:
        with open(filePath, 'r') as f:
            return True
    except FileNotFoundError as e:
        return False
    except IOError as e:
        return False


def create_dir(filePath):
    if not file_exists(filePath):
        os.mkdir(filePath)


def color_list(num_elements):
    random.seed(0)
    clist = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(num_elements)]
    clist[0] = (255, 255, 255)  # BACKGROUND
    clist[1] = (255, 0, 0)  # CANA VERMELHO
    clist[2] = (0, 255, 255)  # ESTILHAÇO
    clist[3] = (255, 51, 255)  # RAIZ
    clist[4] = (0, 255, 0)  # TOCO VERDE
    clist[5] = (0, 0, 255)  # TOLETE AZUL
    return clist


def matrix2augimage(matrix, size_tuple):
    mat = matrix
    mat[mat == 0] = 255
    mat[mat < 255] = 0
    mat = cv2.merge((mat, mat, mat))
    mat = mat.astype(np.uint8)
    img_res = Image.fromarray(mat)
    img_res = img_res.resize(size_tuple, Image.ANTIALIAS)
    return img_res


def save_file(path, name_file, extension, data, num_format):
    if extension == 'txt':
        np.savetxt(path + name_file + "." + extension, data, delimiter='', fmt=num_format)  # '%.4f'
    else:
        if extension == 'csv':
            np.savetxt(path + name_file + "." + extension, data, delimiter=',', fmt=num_format)
        else:
            if extension == 'png':
                save_image(path + name_file + "." + extension, data)


def save_image(name_file, data):
    show_graph = False
    if show_graph:
        width = data.shape[1]
        height = data.shape[0]
        plt.figure(figsize=(width / 1000, height / 1000), dpi=100)
        imgplot = plt.imshow(data)
        imgplot.set_cmap('RdYlGn')
        # min1 = NDVIc[np.isfinite(map)].min()
        # max1 = NDVIc[np.isfinite(map)].max()
        # plt.clim(min1, max1)
        plt.colorbar()
        # plt.axis('off')
        plt.title('NDVIc')
        pylab.show()
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(name_file, dpi=1000)
    else:
        plt.imsave(name_file, data, dpi=1000)
        # plt.imsave(name_file, data, dpi=1000, cmap='RdYlGn')


def roi_extraction(rgb, gt, labels):
    height = rgb.shape[0]
    width = rgb.shape[1]

    # Creating image with only interest regions
    img_roi = np.zeros([height, width, 3], dtype=np.uint8)
    img_roi.fill(255)  # or img[:] = 255

    # Mask file: True = regions of interest
    mask = np.full((height, width), False, dtype=bool)

    for l in labels:

        ##print(">> Obtaining region " + str(l))

        result = np.where(gt == l)
        listOfCoordinates = list(zip(result[0], result[1]))

        for cord in listOfCoordinates:
            b = rgb[cord[0], cord[1], 0]
            g = rgb[cord[0], cord[1], 1]
            r = rgb[cord[0], cord[1], 2]

            img_roi[cord[0], cord[1], 0] = r
            img_roi[cord[0], cord[1], 1] = g
            img_roi[cord[0], cord[1], 2] = b

            mask[cord[0], cord[1]] = True

    return [img_roi, mask]


def iou_metric(gt, pred, num_classes):
    # https://fairyonice.github.io/Learn-about-Fully-Convolutional-Networks-for-semantic-segmentation.html
    # Using Intersection over Union (IoU) measure for each class
    # Average IoU is equal to TP/(FN + TP + FP)
    iou = get_iou(gt, pred, num_classes)
    m_iou = np.mean(iou)
    v_iou = np.var(iou)
    d_iou = np.std(iou)
    return [iou.tolist(), [m_iou, v_iou, d_iou]]


### https://github.com/aleju/imgaug
def data_augmentation(img_path, seg_path):
    seq = iaa.Sequential([
        iaa.Crop(px=(0, 16)),  # crop images from each side by 0 to 16px (randomly chosen)
        iaa.Fliplr(0.5),  # horizontally flip 50% of the images
        iaa.GaussianBlur(sigma=(0, 3.0))  # blur images with a sigma of 0 to 3.0
    ])

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    seg = cv2.cvtColor(cv2.imread(seg_path), cv2.COLOR_BGR2RGB)[:, :, 0]

    aug_det = seq.to_deterministic()
    image_aug = aug_det.augment_image(img)

    segmap = ia.SegmentationMapOnImage(seg, nb_classes=np.max(seg) + 1, shape=img.shape)
    segmap_aug = aug_det.augment_segmentation_maps(segmap)
    segmap_aug = segmap_aug.get_arr_int()

    # image_aug -> IMAGEM -> 1-aug1.png
    # segmap_aug -> ANNOTATION -> 1-aug1.png

    return image_aug, segmap_aug


def data_all(img_route, seg_route, open_name, save_name):
    img, seg = data_augmentation(img_route + open_name, seg_route + open_name)
    image = Image.fromarray(img.astype(np.uint8))
    image.show()
    seg_image = Image.fromarray(seg.astype(np.uint8))
    seg_image.show()
    save_file(img_route, save_name, 'png', image, '')
    save_file(seg_route, save_name, 'png', seg_image, '')

'''
'''im_col = cv2.imread("results/typification/" + result_desc + "_colored_" + f)
   im_col[np.where(im_col == 211)] = 255
   img_hsv = cv2.cvtColor(im_col, cv2.COLOR_RGB2HSV)
   lab = cv2.cvtColor(im_col, cv2.COLOR_BGR2LAB)
   l, a, b = cv2.split(lab)
   clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
   cl = clahe.apply(l)
   limg = cv2.merge((cl, a, b))
   final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)'''

class ST_functions:
    # ------------------------------------------------------------------------------
    # MAIN FUNCTIONS
    # ------------------------------------------------------------------------------

    def __init__(self):
        pass

    def class_error(self, pred, test_labels, classe):
        c = 0
        dflabels = pd.DataFrame(test_labels)
        dfpred = pd.DataFrame(pred)

        ind = dflabels.loc[dflabels[0] == classe].index
        labels_val = dflabels.loc[dflabels[0] == classe].to_numpy()

        pred_val = dfpred.iloc[ind].to_numpy()

        if len(labels_val) == 0 or len(labels_val) == []:
            error = -1
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

    def get_batch_data(self, train_data_path, test_data_path, class_index, join_data, size_batch, iter):
        col1 = np.array([list(range(1, 9))])
        # col2 = np.array([list(range(10,22))])
        col3 = np.array([np.hstack([[0] * 5, [1] * 3])])

        # train_data = np.concatenate((col1, col2), axis=0)
        train_data = np.concatenate((col1, col3), axis=0).T
        train_data = pd.DataFrame(train_data)

        col1 = np.array([list(range(9, 13))])
        # col2 = np.array([list(range(10,16))])
        col3 = np.array([np.hstack([[0] * 3, [1] * 1])])

        # test_data = np.concatenate((col1, col2), axis=0)
        test_data = np.concatenate((col1, col3), axis=0).T
        test_data = pd.DataFrame(test_data)

        df_training = []
        train = []
        train_labels = []
        if train_data_path:
            df_training = pd.read_csv(train_data_path)  # , header=None)
            # print(df_training)
            # df_training = train_data
            feat_index = list(range(df_training.shape[1]))
            feat_index.remove(class_index)
            train = df_training.iloc[:, feat_index].values
            train_labels = df_training.iloc[:, class_index].values

        df_test = []
        test = []
        test_labels = []
        if test_data_path:
            df_test = pd.read_csv(test_data_path)  # , header=None)
            # print(df_test)
            # df_test = test_data
            feat_index = list(range(df_test.shape[1]))
            feat_index.remove(class_index)
            test = df_test.iloc[:, feat_index].values
            test_labels = df_test.iloc[:, class_index].values

        if join_data:
            data = np.concatenate([train, test])
            data_labels = np.concatenate([train_labels, test_labels])
        else:
            data = train
            data_labels = train_labels
        # print(data.shape)
        # print(data_labels)

        num_objects = data.shape[0]

        folds = int(num_objects / size_batch)

        print(folds)

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

        for train_index, test_index in skf.split(data, data_labels):
            if (iter == i):
                # print ("\nIteração = ", i)
                # print("TEST-DATA:", data[test_index], "\nTEST-LABELS:", data_labels[test_index])
                # print("TRAIN-DATA:", data[train_index], "\nTRAIN-LABELS:", data_labels[train_index])
                test_data_fold = data[test_index]
                test_labels_fold = data_labels[test_index]
                train_data_fold = data[train_index]
                train_labels_fold = data_labels[train_index]
            i = i + 1

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

        return test_data_fold, test_labels_fold, train_data_fold, train_labels_fold



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

        matDist = np.array(self.euclidean_distances(data, data))

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
        # print("Selected Objects Classes: " + str(objects_labels.values.ravel()))
        train = pd.DataFrame(train)
        train_labels = pd.DataFrame(train_labels)
        train.columns = objects.columns
        train_labels.columns = objects_labels.columns
        tr = pd.concat([train, objects], axis=0)
        trl = pd.concat([train_labels, objects_labels], axis=0)
        te = test.drop(test.index[sel_objects])
        tel = test_labels.drop(test_labels.index[sel_objects])
        return [tr.to_numpy(), trl.to_numpy(), te.to_numpy(), tel.to_numpy()]


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
                labels[j] = int(np.where(y[j, :] == np.max(y[j, :]))[0])
        return y, labels


    def eds(self, train, test, y, SSet, DistMat):
        ### entropy measuse
        e = self.calc_class_entropy(y)
        candidates = e > np.percentile(e, 75)
        values = np.array(e)[candidates]

        #### density measure - não funciona bem!
        d = self.calc_density(SSet)
        candidates = d > np.percentile(d, 75)
        values = np.array(d)[candidates]

        ### low density measure
        l = self.calc_low_density(DistMat)
        candidates = l > np.percentile(l, 75)
        values = np.array(l)[candidates]

        #### silhouette measure
        from sklearn.metrics import silhouette_samples
        sil_test = np.concatenate([train, test])
        clabels = self.classAnnotation(sil_test)
        sil_values = silhouette_samples(sil_test, clabels[0])
        s = sil_values[len(test) * (-1):]
        candidates = s > np.percentile(s, 25)
        values = np.array(s)[candidates]

        ### ensembles
        el = np.multiply(e, l)
        candidates = el > np.percentile(el, 75)
        values = np.array(el)[candidates]

        sc = 1 - s
        esc = np.multiply(e, sc)
        candidates = esc > np.percentile(esc, 75)
        values = np.array(esc)[candidates]

        return [candidates, values]


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
        color_discrete_map = {'-3': 'rgb(255,255,0)',
                              'Centers': 'rgb(0,0,0)',
                              '-1': 'rgb(255,0,0)',
                              'Cluster 0': 'rgb(0,138,0)',
                              'Cluster 1': 'rgb(168, 2, 2)',
                              'Cluster 2': 'rgb(0,0,133)',
                              'Cluster 3': 'rgb(189,91,0)',
                              'Cluster 4': 'rgb(130,0,156)'}

        df_X = pd.DataFrame({'X': X[:, 0], 'Y': X[:, 1], 'Labels': [str(labels[i]) for i in range(len(labels))]})
        if (med_ind_list != []):
            med_labels = np.array([-1] * len(med_ind_list))
            df_M = pd.DataFrame({'X': X[med_ind_list, 0], 'Y': X[med_ind_list, 1],
                                 'Labels': med_labels.astype(str)})
            df_X = pd.concat([df_X, df_M])

        fig = px.scatter(df_X, x='X', y='Y', color='Labels',
                         color_discrete_map=color_discrete_map, width=500, height=400)
        fig.update_traces(marker=dict(size=8), line=dict(color='rgb(0,0,0)', width=4),
                          selector=dict(mode='Masked'))
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

            a0 = dist_new_obj_center * density_new_objs
            b0 = distance_nearest_data_object * density_data

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
