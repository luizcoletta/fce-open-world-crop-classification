# ------------------------------------------------------------------------------
# USEFUL FUNCTIONS
# > These functions are required in different places in the code
# ------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pylab
import random
import cv2 # OpenCV (Open Source Computer Vision Library) is an open source computer vision and machine learning software library (https://opencv.org/)
from PIL import Image
from keras_segmentation.metrics import get_iou
import os
import imgaug as ia
import imgaug.augmenters as iaa


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
    clist[0] = (255, 255, 255) # BACKGROUND
    clist[1] = (255, 0, 0) # CANA VERMELHO
    clist[2] = (0, 255, 255) # ESTILHAÇO
    clist[3] = (255, 51, 255)# RAIZ
    clist[4] = (0, 255, 0) # TOCO VERDE
    clist[5] = (0, 0, 255) # TOLETE AZUL
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
        np.savetxt(path + name_file + "." + extension, data, delimiter='', fmt=num_format) # '%.4f'
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
        plt.figure(figsize=(width/1000, height/1000), dpi=100)
        imgplot = plt.imshow(data)
        imgplot.set_cmap('RdYlGn')
        #min1 = NDVIc[np.isfinite(map)].min()
        #max1 = NDVIc[np.isfinite(map)].max()
        #plt.clim(min1, max1)
        plt.colorbar()
        #plt.axis('off')
        plt.title('NDVIc')
        pylab.show()
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(name_file, dpi=1000)
    else:
        plt.imsave(name_file, data, dpi=1000)
        #plt.imsave(name_file, data, dpi=1000, cmap='RdYlGn')


def roi_extraction(rgb, gt, labels):

    height = rgb.shape[0]
    width = rgb.shape[1]

    # Creating image with only interest regions
    img_roi = np.zeros([height, width, 3], dtype=np.uint8)
    img_roi.fill(255) # or img[:] = 255

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
    seg = cv2.cvtColor(cv2.imread(seg_path), cv2.COLOR_BGR2RGB)[:,:,0]

    aug_det = seq.to_deterministic()
    image_aug = aug_det.augment_image(img)

    segmap = ia.SegmentationMapOnImage(seg, nb_classes=np.max(seg) + 1, shape=img.shape)
    segmap_aug = aug_det.augment_segmentation_maps(segmap)
    segmap_aug = segmap_aug.get_arr_int()

    return image_aug, segmap_aug

'''im_col = cv2.imread("results/typification/" + result_desc + "_colored_" + f)
   im_col[np.where(im_col == 211)] = 255
   img_hsv = cv2.cvtColor(im_col, cv2.COLOR_RGB2HSV)
   lab = cv2.cvtColor(im_col, cv2.COLOR_BGR2LAB)
   l, a, b = cv2.split(lab)
   clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
   cl = clahe.apply(l)
   limg = cv2.merge((cl, a, b))
   final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)'''
