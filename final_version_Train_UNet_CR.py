
import os
import time
import SimpleITK as sitk
import numpy as np
import scipy
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, ZeroPadding2D, BatchNormalization, Activation
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.preprocessing.image import apply_affine_transform
from keras.callbacks import TensorBoard,ModelCheckpoint,LearningRateScheduler
import warnings
K.set_image_data_format('channels_last')

# set gpu

# 打印可使用设备情况
# from tensorflow.python.client import device_lib
# print('DEVICE LIB LIST LOCAL DEVICES: ',device_lib.list_local_devices())
warnings.filterwarnings("ignore")
import keras.backend.tensorflow_backend as KTF
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5 #设置gpu利用百分比
sess = tf.Session(config=config)
KTF.set_session(sess)

###
rows_standard = 200  # define the input size
cols_standard = 200


def dice_coef_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice_coef=(2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1.0-dice_coef


def conv_bn_relu(nd, k=3, inputs=None):
    conv = Conv2D(nd, k, padding='same')(inputs) #, kernel_initializer='he_normal'
    #bn = BatchNormalization()(conv)
    relu = Activation('relu')(conv)
    return relu

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


#define U-Net architecture
def get_unet(img_shape = None, first5=True):
        inputs = Input(shape = img_shape)
        concat_axis = -1
        if first5: filters = 5
        else: filters = 3

        conv1 = conv_bn_relu(64, filters, inputs)
        conv1 = conv_bn_relu(64, filters, conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = conv_bn_relu(96, 3, pool1)
        conv2 = conv_bn_relu(96, 3, conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = conv_bn_relu(128, 3, pool2)
        conv3 = conv_bn_relu(128, 3, conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = conv_bn_relu(256, 3, pool3)
        conv4 = conv_bn_relu(256, 4, conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = conv_bn_relu(512, 3, pool4)
        conv5 = conv_bn_relu(512, 3, conv5)

        up_conv5 = UpSampling2D(size=(2, 2))(conv5)
        ch, cw = get_crop_shape(conv4, up_conv5)
        crop_conv4 = Cropping2D(cropping=(ch,cw))(conv4)
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
        crop_conv1 = Cropping2D(cropping=(ch,cw))(conv1)
        up9 = concatenate([up_conv8, crop_conv1], axis=concat_axis)
        conv9 = conv_bn_relu(64, 3, up9)
        conv9 = conv_bn_relu(64, 3, conv9)

        ch, cw = get_crop_shape(inputs, conv9)
        conv9 = ZeroPadding2D(padding=(ch, cw))(conv9)
        conv10 = Conv2D(1, 1, activation='sigmoid', padding='same')(conv9) #, kernel_initializer='he_normal'
        model = Model(inputs=inputs, outputs=conv10)
        # model.compile(optimizer=Adam(lr=(2e-4)), loss=dice_coef_loss)
        model.compile(optimizer=Adam(lr=(2e-4)), loss=dice_coef_loss)
        return model


# data augmentation to T2flair, T1, WMH mask
def augmentation(img1, img2, mask):
    """
    :param img1: T2flair
    :param img2: T1
    :param mask: mask
    theta:是用户指定旋转角度范围，其参数只需指定一个整数即可
    tx,ty:分别是水平位置平移和上下位置平移，其参数可以是[0, 1]的浮点数，也可以大于1，其最大平移距离为图片长或宽的尺寸乘以参数
    shear: 是错切变换，效果就是让所有点的x坐标(或者y坐标)保持不变，而对应的y坐标(或者x坐标)则按比例发生平移，且平移的大小和该点到x轴(或y轴)的垂直距离成正比。
    zx: Zoom in x direction.
    zy: Zoom in y direction
    row_axis: Index of axis for rows in the input image.
    col_axis: Index of axis for columns in the input image.
    channel_axis: Index of axis for channels in the input image.
    cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    fill_mode:为填充模式，如前面提到，当对图片进行平移、放缩、错切等操作时，图片中会出现一些缺失的地方，那这些缺失的地方该用什么方式补全呢？就由fill_mode中的参数确定，包括：“constant”、“nearest”（默认）、“reflect”和“wrap”。
    :return:
    """
    # img1
    img1 = apply_affine_transform(img1[..., np.newaxis], theta=5, tx=0, ty=0, shear=0.05, zx=1, zy=1,
                           row_axis=0, col_axis=1, channel_axis=2,fill_mode='nearest', cval=0., order=1)
    img1 = img1[:,0,:]
    # img2
    img2 = apply_affine_transform(img2[..., np.newaxis], theta=5, tx=0, ty=0, shear=0.05, zx=1, zy=1,
                                  row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0., order=1)
    img2 = img2[:,0,:]
    # mask
    mask = apply_affine_transform(mask[..., np.newaxis], theta=5, tx=0, ty=0, shear=0.05, zx=1, zy=1,
                                  row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0., order=1)
    mask = mask[:,0,:]
    return img1, img2, mask


###----define prepocessing methods/tricks for different datasets 对 mask 同样进行前期处理，高斯归一化，维度标准化，同 Flair ,T1 一样------------------------
def Utrecht_preprocessing(FLAIR_image, T1_image):
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
    brain_mask_FLAIR[FLAIR_image < thresh_FLAIR] = 0 # 同理，找到FLAIR_image数据中小于规定阈值的索引，按照其索引把brain_mask_FLAIR相应置0
    """
   ndimage.binary_fill_holes()可以填充孔洞,类似闭运算，但更准确
   腐蚀算法（像素集合向内缩小一个像素区域）；
   膨胀算法（像素向周围延展一个像素））；
   开运算也就是先腐蚀再膨胀的过程，平滑较大物体的边界的同时并不明显改变其面积。
   闭运算是先膨胀再腐蚀的运算。闭运算可以用来填充物体内细小空洞、连接邻近物体、平滑其边界的同时并不明显改变其面积。
    """
    # fill the holes inside brain
    for iii in range(np.shape(FLAIR_image)[0]):
        brain_mask_FLAIR[iii, :, :] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_FLAIR[iii,:,:])

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
    # T1  ------------------------------------------
    brain_mask_T1[T1_image >=thresh_T1] = 1
    brain_mask_T1[T1_image < thresh_T1] = 0
    for iii in range(np.shape(T1_image)[0]):
        brain_mask_T1[iii, :, :] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_T1[iii,:,:])  #fill the holes inside brain

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

# pre process data: mask
def Utrecht_mask_preprocessing(mask_image):
    num_selected_slice = np.shape(mask_image)[0]
    image_rows_Dataset = np.shape(mask_image)[1]
    image_cols_Dataset = np.shape(mask_image)[2]
    mask_image = np.float32(mask_image)

    # crop standard size
    def crop_imaging(imaging):
        imaging = imaging[:,
                  int((image_rows_Dataset - rows_standard) / 2):int((image_rows_Dataset + rows_standard) / 2),
                  int(image_cols_Dataset / 2 - cols_standard / 2):int(image_cols_Dataset / 2 + cols_standard / 2)]
        return imaging

    # crop imaging to standard
    mask_image = crop_imaging(mask_image)
    return mask_image

def GE3T_preprocessing(FLAIR_image, T1_image):
  #  start_slice = 10
    thresh_FLAIR = 70
    thresh_T1 = 30
    start_cut = 46
    num_selected_slice = np.shape(FLAIR_image)[0]
    image_rows_Dataset = np.shape(FLAIR_image)[1]
    image_cols_Dataset = np.shape(FLAIR_image)[2]

    FLAIR_image = np.float32(FLAIR_image)
    T1_image = np.float32(T1_image)
    # initialize some array
    brain_mask_FLAIR = np.ndarray((np.shape(FLAIR_image)[0],image_rows_Dataset, image_cols_Dataset), dtype=np.float32)
    brain_mask_T1 = np.ndarray((np.shape(FLAIR_image)[0],image_rows_Dataset, image_cols_Dataset), dtype=np.float32)
    FLAIR_image_suitable = np.ndarray((num_selected_slice, rows_standard, cols_standard), dtype=np.float32)
    T1_image_suitable = np.ndarray((num_selected_slice, rows_standard, cols_standard), dtype=np.float32)

    # FLAIR --------------------------------------------
    brain_mask_FLAIR[FLAIR_image >=thresh_FLAIR] = 1
    brain_mask_FLAIR[FLAIR_image < thresh_FLAIR] = 0
    for iii in range(np.shape(FLAIR_image)[0]):
        brain_mask_FLAIR[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_FLAIR[iii,:,:])  #fill the holes inside brain

    #------Gaussion Normalization
    FLAIR_image -=np.mean(FLAIR_image[brain_mask_FLAIR == 1])
    FLAIR_image /=np.std(FLAIR_image[brain_mask_FLAIR == 1])
    # FLAIR padded
    FLAIR_image_suitable[...] = np.min(FLAIR_image)
    FLAIR_image_suitable[:, :,
    int(cols_standard / 2 - image_cols_Dataset / 2):int(cols_standard / 2 + image_cols_Dataset / 2)] = \
        FLAIR_image[:, start_cut:start_cut + rows_standard, :]

    # T1 -----------------------------------------------
    brain_mask_T1[T1_image >=thresh_T1] = 1
    brain_mask_T1[T1_image < thresh_T1] = 0
    for iii in range(np.shape(T1_image)[0]):
        brain_mask_T1[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_T1[iii,:,:])  #fill the holes inside brain
    #------Gaussion Normalization
    T1_image -=np.mean(T1_image[brain_mask_T1 == 1])
    T1_image /=np.std(T1_image[brain_mask_T1 == 1])

    # T1 padded
    T1_image_suitable[...] = np.min(T1_image)
    T1_image_suitable[:, :,
    int(cols_standard / 2 - image_cols_Dataset / 2):int(cols_standard / 2 + image_cols_Dataset / 2)] = \
        T1_image[:, start_cut:start_cut + rows_standard, :]

    # FLAIR T1 concatenate---------------------------------------------------
    FLAIR_image_suitable  = FLAIR_image_suitable[..., np.newaxis]
    T1_image_suitable  = T1_image_suitable[..., np.newaxis]

    imgs_two_channels = np.concatenate((FLAIR_image_suitable, T1_image_suitable), axis = 3)
    return imgs_two_channels

def GE3T_mask_preprocessing(mask_image):
    num_selected_slice = np.shape(mask_image)[0]
    image_rows_Dataset = np.shape(mask_image)[1]
    image_cols_Dataset = np.shape(mask_image)[2]
    mask_image = np.float32(mask_image)

    start_cut = 46
    # initialize some array
    image_suitable = np.ndarray((num_selected_slice, rows_standard, cols_standard), dtype=np.float32)
    # imaging padded
    image_suitable[...] = np.min(mask_image)
    image_suitable[:, :,
    int(cols_standard / 2 - image_cols_Dataset / 2):int(cols_standard / 2 + image_cols_Dataset / 2)] = \
        mask_image[:, start_cut:start_cut + rows_standard, :]
    return image_suitable


# 对数据slice进行删除，根据经验选择
def slice_delect(imaging):
    imaging_array_axis = np.shape(imaging)
    if imaging_array_axis[0] == 48:
        imaging = imaging[5:-5, :, :]
    if imaging_array_axis[0] == 83:
        imaging = imaging[12:-8, :, :]
    return imaging


def data_process(inputDir=None,outputDir=None):
    # 定义后面合并数据初始数据数组
    T2flair_T1_concate_Total = np.zeros((1, 200, 200, 2), dtype=np.float32)
    mask_concate_Total = np.zeros((1, 200, 200, 1), dtype=np.float32)

    # 数据根目录
    preparentfolders = os.listdir(inputDir)
    # 对乱序文件夹进行重新排序
    preparentfolders.sort(key=lambda  x: int(x.split('.')[0]))

    count_num = 0
    for subfolders1 in preparentfolders:
        count_num += 1
        # print()
        print('****** Loading data and  Processing  data******')
        print('Count Num: ', count_num)

        path_100 = os.path.join(inputDir, subfolders1)
        path_100_subfiles = os.listdir(path_100)
        if path_100_subfiles[0] == 'pre':
            path_pre = os.path.join(path_100, path_100_subfiles[0])
            path_pre_subfiles = os.listdir(path_pre)

            # T2FLAIR
            path_T2flair = os.path.join(path_pre, path_pre_subfiles[0])
            # print('path_T2flair: ', path_T2flair)
            T2flair_image = sitk.ReadImage(path_T2flair)
            T2flair_array = sitk.GetArrayFromImage(T2flair_image)

            # T1
            path_T1 = os.path.join(path_pre, path_pre_subfiles[1])
            # print('path_T1: ', path_T1)
            T1_image = sitk.ReadImage(path_T1)
            T1_array = sitk.GetArrayFromImage(T1_image)

            # preprocess T2flair T1
            if count_num <= 40:
                imgs_T2flair_T1_single_patient = Utrecht_preprocessing(T2flair_array,T1_array)
            else:
                imgs_T2flair_T1_single_patient = GE3T_preprocessing(T2flair_array, T1_array)

            # crop and padded to standard with 200*200,select slice
            imgs_T2flair_T1_single_patient_delect = slice_delect(imgs_T2flair_T1_single_patient)

            # imgs_T2flair_T1_concate = np.concatenate((T2flair_array, T1_array),axis=0)
            # save T2,T1,
            filename_resultImage = './Data_Select_pre_T2Flair_T1_mask_sort/raw_sorted_crop_norm_delect/images/'+str(count_num-1)+'.nii.gz'
            sitk.WriteImage(sitk.GetImageFromArray(imgs_T2flair_T1_single_patient_delect), filename_resultImage)

        if path_100_subfiles[1] == 'wmh.nii.gz':
            path_wmh = os.path.join(path_100, path_100_subfiles[1])
            # print('path_wmh: ', path_wmh)
            mask_image = sitk.ReadImage(path_wmh)
            mask_array = sitk.GetArrayFromImage(mask_image)

            if count_num <= 40:
                mask_array = Utrecht_mask_preprocessing(mask_array)
            else:
                mask_array = GE3T_mask_preprocessing(mask_array)

            # crop and padded to standard with 200*200
            imgs_mask_delect = slice_delect(mask_array)
            imgs_mask_delect = imgs_mask_delect[...,np.newaxis]

            filename_resultImage = './Data_Select_pre_T2Flair_T1_mask_sort/raw_sorted_crop_norm_delect/mask' + str(
                count_num - 1) + '.nii.gz'
            sitk.WriteImage(sitk.GetImageFromArray(imgs_mask_delect), filename_resultImage)

        T2flair_T1_concate_Total = np.concatenate((T2flair_T1_concate_Total,imgs_T2flair_T1_single_patient_delect),axis=0)
        mask_concate_Total = np.concatenate((mask_concate_Total,imgs_mask_delect),axis=0)

        # 将数据最后保存为 npy 文件
    # np.save('./imgs_T2flair_T1_concate_Total_npy.npy',T2flair_T1_concate_Total)
    # np.save('./imgs_mask_concate_Total_npy.npy',mask_concate_Total)

    return T2flair_T1_concate_Total[49:, ...], mask_concate_Total[49:,...]


#train single model on the training set
def train_leave_one_out(images, masks, patient=0, flair=True, t1=True, epoch_total=80,first5=True, aug=True, verbose=False):
    """if patient < 40:
        images = np.delete(images, range(patient * 38, (patient + 1) * 38), axis=0)
        masks = np.delete(masks, range(patient * 38, (patient + 1) * 38), axis=0)
    else:
        images = np.delete(images, range(1520 + (patient - 40) * 63, 1520 + (patient - 39) * 63), axis=0)
        masks = np.delete(masks, range(1520 + (patient - 40) * 63, 1520 + (patient - 39) * 63), axis=0)"""

    # 对数据进行 inverse copy 扩增
    if aug:
        images = np.concatenate((images, images[..., ::-1, :]), axis=0)
        masks = np.concatenate((masks, masks[..., ::-1, :]), axis=0)

    samples_num = images.shape[0]
    row = images.shape[1]
    col = images.shape[2]

    """Super parameter"""
    epoch_total = epoch_total   # train epoch
    batch_size = 32   # 训练每个批次中样本量
    img_shape = (row, col, flair+t1)   # 将 input 数据 shape ，传输到get_unet中
    model = get_unet(img_shape, first5)

    current_epoch = 1
    while current_epoch <= epoch_total:
        print('Epoch ', str(current_epoch), '/', str(epoch_total))
        if aug:
            images_aug = np.zeros(images.shape, dtype=np.float32)
            masks_aug = np.zeros(masks.shape, dtype=np.float32)
            for i in range(samples_num):
                images_aug[i, ..., 0], images_aug[i, ..., 1], masks_aug[i, ..., 0] = augmentation(images[i, ..., 0], images[i, ..., 1], masks[i, ..., 0])
            #  合并扩增后的的数据
            image = np.concatenate((images, images_aug), axis=0)
            mask = np.concatenate((masks, masks_aug), axis=0)
        else:
            image = images.copy()
            mask = masks.copy()

        # callback_lists
        def scheduler(epoch):
            # step
            if epoch % 5 == 0 and epoch != 0:
                lr = K.get_value(model.optimizer.lr)
                K.set_value(model.optimizer.lr, lr * 0.1)
                print("lr changed to {}".format(lr * 0.1))
            return K.get_value(model.optimizer.lr)

        tensorboard = TensorBoard(log_dir='./log')  # log
        checkpoint_path = "./models/our/best_weights.h5"
        checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor="loss", mode='min', save_weights_only=True,
                                     save_best_only=True, verbose=1, period=1)
        reduce_lr = LearningRateScheduler(scheduler)
        callback_lists = [tensorboard, checkpoint]  # ,reduce_lr

        # 将数据送到网络中，开始训练
        history = model.fit(image, mask, batch_size=batch_size, epochs=1, verbose=verbose, shuffle=True,callbacks=callback_lists)
        current_epoch += 1
        if history.history['loss'][-1] > 0.99:
            model = get_unet(img_shape, first5)
            current_epoch = 1
        print('Dice Coefficient: ', 1.0 - history.history['loss'][-1])

    # model 保存的路径
    model_path = 'models/'
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    model_path += 'our/'
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    model_path += str(patient)
    model_path += '_'

    # save weighted
    model_h5 = model_path + 'four_json.h5'
    model.save_weights(model_h5)
    print('Model saved to ', model_h5)

    # save json
    path_model_json = model_path + 'four.json'
    json_string = model.to_json()
    open(path_model_json, 'w').write(json_string)
    print('Model saved to ', path_model_json)

    # save json and weight
    path_model_json_weight = model_path + 'four_json_weight.hdf5'
    model.save(path_model_json_weight)
    print('Model saved to ', path_model_json_weight)


#leave-one-out evaluation
def main():
    # time start
    start_time = time.time()
    warnings.filterwarnings("ignore")
    # 对三组数据进行重新排序，并对其slice进行丢弃操作，48->38,两组20; 83->63,一组20，同时将处理后的数据保存为 npy 数据格式
    inputDir = './Data_Select_pre_T2Flair_T1_mask_sort/raw_sorted'
    outputDir = './Data_Select_pre_T2Flair_T1_mask_sort/raw_sorted_npy'
    # crop , Gaussion Normalization,select slice
    # images,masks = data_process(inputDir=inputDir,outputDir=outputDir)

    images = np.load('./imgs_T2flair_T1_concate_Total_npy.npy')
    masks = np.load('./imgs_mask_concate_Total_npy.npy')

    model_num = 3
    for num in range(0, model_num):
        print('Patient Index Delete: ', str(num), " Processing ")
        train_leave_one_out(images, masks, verbose=True)
    # time stop
    stop_time = time.time()
    process_time = start_time-stop_time
    print('Process Time: ',process_time/3600,('h'))
if __name__=='__main__':
    main()

