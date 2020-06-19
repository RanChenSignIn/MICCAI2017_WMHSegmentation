from __future__ import print_function
import os
import numpy as np
import SimpleITK as sitk
from keras.layers.merge import concatenate
import scipy.spatial
from keras.models import Model
from keras.optimizers import Adam
from evaluation import getDSC, getHausdorff, getLesionDetection, getAVD, getImages
from keras import backend as K
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, ZeroPadding2D, Activation
import shutil

import warnings
warnings.filterwarnings("ignore")


## some pr-edefined parameters
rows_standard = 200  #the input size
cols_standard = 200
thresh = 30   # threshold for getting the brain mask


# COPY DATA TO ONE FOLDERS
def copy_data(inputDir=None,outputDir=None):
    # 数据根目录  shutil.copy(sourceDir,  targetDir)
    preparentfolders = os.listdir(inputDir)
    # 对乱序文件夹进行重新排序
    preparentfolders.sort(key=lambda  x: int(x.split('.')[0]))
    count_num = 0
    for subfolders1 in preparentfolders:
        # creat folders
        # pathoutnum = os.path.join(outputDir,subfolders1)
        pathoutnum = outputDir
        if not os.path.exists(pathoutnum):
            os.mkdir(pathoutnum)
        count_num += 1
        print('****** Copy data******')
        print('Count Num: ', count_num)

        path_100 = os.path.join(inputDir, subfolders1)
        path_100_subfiles = os.listdir(path_100)
        if path_100_subfiles[0] == 'pre':
            path_pre = os.path.join(path_100, path_100_subfiles[0])
            path_pre_subfiles = os.listdir(path_pre)

            # T2FLAIR
            path_T2flair = os.path.join(path_pre, path_pre_subfiles[0])
            shutil.copy(path_T2flair, pathoutnum)
            # T1
            path_T1 = os.path.join(path_pre, path_pre_subfiles[1])
            shutil.copy(path_T2flair, pathoutnum)

        if path_100_subfiles[1] == 'wmh.nii.gz':
            path_wmh = os.path.join(path_100, path_100_subfiles[1])
            shutil.copy(path_wmh, pathoutnum)


# -define u-net architecture--------------------
def dice_coef_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dece_coef = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1.0-dece_coef

# U-Net concate
def get_crop_shape(target, refer):
        # width, the 3rd dimension
        cw = (target.get_shape()[2] - refer.get_shape()[2]).value
        assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 = int(cw/2), int(cw/2) + 1
        else:
            cw1, cw2 = int(cw/2), int(cw/2)
        # height, the 2nd dimension
        ch = (target.get_shape()[1] - refer.get_shape()[1]).value
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = int(ch/2), int(ch/2) + 1
        else:
            ch1, ch2 = int(ch/2), int(ch/2)
        return (ch1, ch2), (cw1, cw2)


# conv function
def conv_bn_relu(nd, k=None, inputs=None):
    conv = Conv2D(nd, k, padding='same')(inputs) #, kernel_initializer='he_normal'
    relu = Activation('relu')(conv)
    return relu


#define U-Net architecture
def get_unet(img_shape = None):
    # define some parameter
    filters1 = 5 # 输入层卷积核5*5
    filters = 3 # 神经网络中间的隐藏层卷积核3*3
    concat_axis = -1 # 全卷机网络在中间有skip concate ，拼接维度为最后一维度

    inputs = Input(shape=img_shape)
    conv1 = conv_bn_relu(64, filters1, inputs)
    conv1 = conv_bn_relu(64, filters1, conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = conv_bn_relu(96, filters, pool1)
    conv2 = conv_bn_relu(96, filters, conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv_bn_relu(128, filters, pool2)
    conv3 = conv_bn_relu(128, filters, conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = conv_bn_relu(256, filters, pool3)
    conv4 = conv_bn_relu(256, 4, conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = conv_bn_relu(512, filters, pool4)
    conv5 = conv_bn_relu(512, filters, conv5)

    up_conv5 = UpSampling2D(size=(2, 2))(conv5)
    ch, cw = get_crop_shape(conv4, up_conv5)  # 得出shape维度差值的一半，width:[1],height:[2],
    crop_conv4 = Cropping2D(cropping=(ch,cw))(conv4)  # 空间尺寸，高度和宽度进行裁剪,
    up6 = concatenate([up_conv5, crop_conv4], axis=concat_axis)
    conv6 = conv_bn_relu(256, 3, up6)
    conv6 = conv_bn_relu(256, 3, conv6)

    up_conv6 = UpSampling2D(size=(2, 2))(conv6)
    ch, cw = get_crop_shape(conv3, up_conv6)
    crop_conv3 = Cropping2D(cropping=(ch,cw))(conv3)
    up7 = concatenate([up_conv6, crop_conv3], axis=concat_axis)
    conv7 = conv_bn_relu(128, 3, up7)
    conv7 = conv_bn_relu(128, 3, conv7)

    up_conv7 = UpSampling2D(size=(2, 2))(conv7)
    ch, cw = get_crop_shape(conv2, up_conv7)
    crop_conv2 = Cropping2D(cropping=(ch,cw))(conv2)
    up8 = concatenate([up_conv7, crop_conv2], axis=concat_axis)
    conv8 = conv_bn_relu(96, 3, up8)
    conv8 = conv_bn_relu(96, 3, conv8)

    up_conv8 = UpSampling2D(size=(2, 2))(conv8)
    ch, cw = get_crop_shape(conv1, up_conv8)
    crop_conv1 = Cropping2D(cropping=(ch, cw))(conv1)
    up9 = concatenate([up_conv8, crop_conv1], axis=concat_axis)
    conv9 = conv_bn_relu(64, 3, up9)
    conv9 = conv_bn_relu(64, 3, conv9)

    ch, cw = get_crop_shape(inputs, conv9) # input shape:(None,200,200,2);conv9 shape(None,192,192,64)
    conv9 = ZeroPadding2D(padding=(ch, cw))(conv9) # 补零   conv9 shape(None,200,200,64)
    conv10 = Conv2D(1, 1, activation='sigmoid', padding='same')(conv9) # 全连接，卷积核1*1，shape(None,200,200,1) kernel_initializer='he_normal'
    # define input output
    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer=Adam(lr=(2e-4)), loss=dice_coef_loss)
    return model


#--------------------------------------------------------------------------------------
def preprocessing(FLAIR_image, T1_image):
    thresh_FLAIR = 70
    thresh_T1 = 30
    channel_num = 2
    # print(np.shape(FLAIR_image))
    num_selected_slice = np.shape(FLAIR_image)[0]
    image_rows_Dataset = np.shape(FLAIR_image)[1]
    image_cols_Dataset = np.shape(FLAIR_image)[2]
    T1_image = np.float32(T1_image)

    # 按照FLAIR_image的数据维度大小，生成数组数据brain_mask_FLAIR,brain_mask_T1，类型为float32
    brain_mask_FLAIR = np.ndarray((np.shape(FLAIR_image)[0],image_rows_Dataset, image_cols_Dataset), dtype=np.float32)
    brain_mask_T1 = np.ndarray((np.shape(FLAIR_image)[0],image_rows_Dataset, image_cols_Dataset), dtype=np.float32)

    # FLAIR --------------------------------------------
    brain_mask_FLAIR[FLAIR_image >=thresh_FLAIR] = 1  # 找到FLAIR_image数据中大于等于规定阈值的索引，按照其索引把brain_mask_FLAIR相应置1，
    brain_mask_FLAIR[FLAIR_image < thresh_FLAIR] = 0  # 同理，找到FLAIR_image数据中小于规定阈值的索引，按照其索引把brain_mask_FLAIR相应置0
    """
           ndimage.binary_fill_holes()可以填充孔洞,类似闭运算，但更准确
           腐蚀算法（像素集合向内缩小一个像素区域）；
           膨胀算法（像素向周围延展一个像素））；
           开运算也就是先腐蚀再膨胀的过程，平滑较大物体的边界的同时并不明显改变其面积。
           闭运算是先膨胀再腐蚀的运算。闭运算可以用来填充物体内细小空洞、连接邻近物体、平滑其边界的同时并不明显改变其面积。
           """
    # fill the holes inside brain
    for iii in range(np.shape(FLAIR_image)[0]):
        brain_mask_FLAIR[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_FLAIR[iii, :, :])

    # crop standard size ,将数据大小裁剪为标准大小（此处规定标准200*200），
    def crop_imaging(imaging):
        imaging = imaging[:,
                  int((image_rows_Dataset - rows_standard) / 2):int((image_rows_Dataset + rows_standard) / 2),
                  int(image_cols_Dataset / 2 - cols_standard / 2):int(image_cols_Dataset / 2 + cols_standard / 2)]
        return imaging

    ###------Gaussion Normalization here
    FLAIR_image -=np.mean(FLAIR_image[brain_mask_FLAIR == 1])  # 把所有高于阈值的体素减去其所有值均值
    FLAIR_image /=np.std(FLAIR_image[brain_mask_FLAIR == 1])  # 把所有高于阈值的体素除去其所有值标准差

    # crop imaging to standard
    FLAIR_image = crop_imaging(FLAIR_image)

    # 双通道导入，则同理将T1 进行高斯归一化，以及维度标准化
    # T1  -----------------------------------------------
    brain_mask_T1[T1_image >= thresh_T1] = 1
    brain_mask_T1[T1_image < thresh_T1] = 0
    for iii in range(np.shape(T1_image)[0]):
        brain_mask_T1[iii, :, :] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_T1[iii, :, :])  #fill the holes inside brain

    #------Gaussion Normalization
    T1_image -=np.mean(T1_image[brain_mask_T1 == 1])
    T1_image /=np.std(T1_image[brain_mask_T1 == 1])

    # crop imaging to standard
    T1_image = crop_imaging(T1_image)

    #---------------------------------------------------
    FLAIR_image  = FLAIR_image[..., np.newaxis]
    T1_image  = T1_image[..., np.newaxis]
    imgs_two_channels = np.concatenate((FLAIR_image, T1_image), axis = 3)
    return imgs_two_channels


def postprocessing(FLAIR_array, pred):
    # 尺寸还原为剪裁前的尺寸
    num_o = np.shape(FLAIR_array)[1]  # original size
    rows_o = np.shape(FLAIR_array)[1]
    cols_o = np.shape(FLAIR_array)[2]
    original_pred = np.zeros(np.shape(FLAIR_array), dtype=np.float32)
    original_pred[:,
                int((rows_o - rows_standard) / 2):int((rows_o + rows_standard) / 2),
                int(cols_o / 2 - cols_standard / 2):int(cols_o / 2 + cols_standard / 2)] = pred[:,:,:,0]
    return original_pred


def main():
    # 对三组数据 copy to predict
    if_copy = True
    if if_copy:
        inputDir = './Data_Select_pre_T2Flair_T1_mask_sort/raw_sorted'
        outputDir = './Data_Select_pre_T2Flair_T1_mask_sort/raw_nopre/'
        copy_data(inputDir=inputDir, outputDir=outputDir)

    # data path
    TEST_INDEX = 0
    inputDir = 'input_dir'
    inputDir = os.path.join(inputDir, str(TEST_INDEX))
    outputDir = 'result'
    if not os.path.exists(outputDir):
        os.mkdir(outputDir)

    if_test = False
    if if_test:
        #Read data----------------------------------------------------------------------------
        # FLAIR
        FLAIR_image = sitk.ReadImage(os.path.join(inputDir, 'FLAIR.nii.gz'))
        FLAIR_array = sitk.GetArrayFromImage(FLAIR_image)
        # T1
        T1_image = sitk.ReadImage(os.path.join(inputDir, 'T1.nii.gz'))
        T1_array = sitk.GetArrayFromImage(T1_image)
        imgs_test = preprocessing(np.float32(FLAIR_array), np.float32(T1_array))  # data preprocessing

        # Load model  , ensemble models
        img_shape = (rows_standard, cols_standard, 2)
        model = get_unet(img_shape)

        # weights path
        model_dir = './models/our/best_weights_models/'

        # Load weights
        model.load_weights(os.path.join(model_dir, '0_four_json.h5'))
        pred_0 = model.predict(imgs_test, batch_size=1, verbose=1)

        model.load_weights(os.path.join(model_dir, '1_four_json.h5'))
        pred_1 = model.predict(imgs_test, batch_size=1, verbose=1)

        model.load_weights(os.path.join(model_dir, '2_four_json.h5'))
        pred_2 = model.predict(imgs_test, batch_size=1, verbose=1)

        model.load_weights(os.path.join(model_dir, '3_four_json.h5'))
        pred_3 = model.predict(imgs_test, batch_size=1, verbose=1)

        # select predict model weight count
        pred_model_count = 4 # 1,2,3,4
        print('Predict Model Weight Count: {}'.format(pred_model_count))
        if pred_model_count == 1:
            pred = pred_0
        elif pred_model_count == 2:
            pred = (pred_0+pred_1) / 2
        elif pred_model_count == 3:
            pred = (pred_0+pred_1 + pred_2) / 3
        else:
            pred = (pred_0 + pred_1 + pred_2 + pred_3) / 4

        pred[pred[...,0] > 0.45] = 1      # 0.45 thresholding
        pred[pred[...,0] <= 0.45] = 0

        # get the original size to match
        original_pred = postprocessing(FLAIR_array, pred)

        # Save predict mask data to .nii.gz
        filename_resultImage = os.path.join(outputDir,'preidict.nii.gz')
        sitk.WriteImage(sitk.GetImageFromArray(original_pred), filename_resultImage )

        compute_metric = True
        if compute_metric:
            filename_testImage = os.path.join(inputDir + '/wmh.nii.gz')
            testImage, resultImage = getImages(filename_testImage, filename_resultImage)
            dsc = getDSC(testImage, resultImage)
            avd = getAVD(testImage, resultImage)
            recall, f1 = getLesionDetection(testImage, resultImage)
            print('Result of prediction:')
            print('Dice',                dsc,       ('higher is better, max=1'))
            print('AVD',                 avd,  '%',  '(lower is better, min=0)')
            print('Lesion detection', recall,       '(higher is better, max=1)')
            print('Lesion F1',            f1,       '(higher is better, max=1)')


if __name__ == '__main__':
    main()














