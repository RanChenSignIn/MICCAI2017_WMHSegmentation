from __future__ import print_function
import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import cv2

from keras.layers.merge import concatenate
import scipy.spatial
from keras.models import Model
from keras.optimizers import Adam
from evaluation import getDSC, getHausdorff, getLesionDetection, getAVD, getImages
from keras import backend as K

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, ZeroPadding2D, Activation
import shutil

import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import warnings

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5 #设置gpu利用百分比
sess = tf.Session(config=config)
KTF.set_session(sess)


# COPY DATA TO ONE FOLDERS
def copy_data(inputDir=None, outputDir=None):
    # concatation
    def concatepreprocessing(path_T2flair, path_T1):
        # T2flair---------------------------------------------------
        T2flair_image = sitk.ReadImage(path_T2flair)
        T2flair_array = sitk.GetArrayFromImage(T2flair_image)
        # T1---------------------------------------------------
        T1_image = sitk.ReadImage(path_T1)
        T1_array = sitk.GetArrayFromImage(T1_image)
        # concate---------------------------------------------------
        FLAIR_image = T2flair_array[..., np.newaxis]
        T1_image = T1_array[..., np.newaxis]
        imgs_two_channels = np.concatenate((FLAIR_image, T1_image), axis=3)
        return imgs_two_channels
    # 数据根目录  shutil.copy(sourceDir,  targetDir)
    preparentfolders = os.listdir(inputDir)
    # 对乱序文件夹进行重新排序
    preparentfolders.sort(key=lambda x: int(x.split('.')[0]))

    count_num_FLAIR = 0
    count_num_wmh = 0
    
    for subfolders1 in preparentfolders:
        # creat folders
        # pathoutnum = os.path.join(outputDir,subfolders1)
        pathoutnum = outputDir
        if not os.path.exists(pathoutnum):
            os.mkdir(pathoutnum)

        print('****** Copy data******')
        print('Count Num: ', count_num_FLAIR)

        path_100 = os.path.join(inputDir, subfolders1)
        path_100_subfiles = os.listdir(path_100)

        # T2FLAIR ; T1 ---------------------------------------------------
        if path_100_subfiles[0] == 'pre':
            path_pre = os.path.join(path_100, path_100_subfiles[0])
            path_pre_subfiles = os.listdir(path_pre)

            # T2FLAIR
            path_T2flair = os.path.join(path_pre, path_pre_subfiles[0])
            # shutil.copy(path_T2flair, pathoutnum)
            # T1
            path_T1 = os.path.join(path_pre, path_pre_subfiles[1])
            T2flair_T1 = concatepreprocessing(path_T2flair, path_T1)

            # save T2,T1,
            slice_count=np.shape(T2flair_T1)[0]
            for k in range(slice_count):
                T2flair_T1_s = T2flair_T1[k, ...]
                filename_resultImage = outputDir + str(count_num_FLAIR ) + '.nii.gz'

                plt.savefig()
                sitk.WriteImage(sitk.GetImageFromArray(T2flair_T1_s), filename_resultImage)
                count_num_FLAIR += 1
        # wmh---------------------------------------------------
        if path_100_subfiles[1] == 'wmh.nii.gz':
            path_wmh = os.path.join(path_100, path_100_subfiles[1])
            wmh_image = sitk.ReadImage(path_wmh)
            wmh_array = sitk.GetArrayFromImage(wmh_image)
            # save wmh
            slice_count = np.shape(wmh_array)[0]
            for k in range(slice_count):
                wmh_array_s = wmh_array[k, ...]
                filename_resultImage = outputDir + str(count_num_wmh) + 'wmh.nii.gz'
                sitk.WriteImage(sitk.GetImageFromArray(wmh_array_s), filename_resultImage)
                count_num_wmh += 1


def main():
    # 对三组数据 copy to predict
    if_copy = True
    if if_copy:
        inputDir = './Data_Select_pre_T2Flair_T1_mask_sort/raw_sorted'
        outputDir = './Data_Select_pre_T2Flair_T1_mask_sort/raw_nopre_concateT2T1/'
        copy_data(inputDir=inputDir, outputDir=outputDir)


if __name__ == '__main__':
    main()














