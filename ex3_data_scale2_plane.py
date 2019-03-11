# common
import numpy as np
import csv
import shutil
import os
import sys
import argparse
import random
# keras
import keras
from keras import Input
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Reshape, Permute
from keras.layers import Dropout, BatchNormalization, Concatenate, UpSampling2D, Conv2DTranspose, ZeroPadding2D
from keras.layers.core import Activation
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16
# tensorflow
import tensorflow as tf
import tensorflow.keras.backend as K
K.set_epsilon(1e-07)
from keras.utils import np_utils
#
import coco
config = coco.CocoConfig()
#
from PIL import Image
#
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/.')
# User func
from Operation.common import *
import Operation.img_to_html as op_img_to_html
import Operation.push_slack as op_push_slack


# papath
path_the_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'
PROGRAM_NAME = os.path.basename(__file__).replace(".py", "").replace("train_", "")
#
TRAIN_FILE_PATH_BASE = '/home/nvidia/Public/depth/data/'
TRAIN_FILE_PATH_SELECT = 'imageDepthOverlap_partial_amax5.csv'
GET_TRAIN_FILE_PATH = TRAIN_FILE_PATH_BASE + TRAIN_FILE_PATH_SELECT
#
HTML_PATH_BASE = path_the_directory + 'predict_image/'
#
COCO_DIR = '/home/nvidia/Public/depth/data/nyu/train'


# number
IMAGE_HEIGHT = 228#240#228
IMAGE_WIDTH = 304#320#304
TARGET_HEIGHT = 55#56#55
TARGET_WIDTH = 74#76#74
CROP_HEIGHT = 56
CROP_WIDTH = 76
SCALE1_OUTPUT_HEIGHT = 55
SCALE1_OUTPUT_WIDTH = 74
SCALE2_INPUT_HEIGHT = 55
SCALE2_INPUT_WIDTH = 74
OVERLAP_HEIGHT = SCALE2_INPUT_HEIGHT
OVERLAP_WIDTH = SCALE2_INPUT_WIDTH
SegNet_OUTPUT_HEIGHT = OVERLAP_HEIGHT
SegNet_OUTPUT_WIDTH = OVERLAP_WIDTH
#
HTML_TIME = 8
OVERLAP_NUMBER = 5 + 1#52
LABEL_NUMBER = LABEL_NUMBER_COMMON
#
KEEP_CON = 0.5
#
STAGE_ALL = STAGE_ALL_COMMON
SCALE_ALL = 2
#
OVERLAP_NORMLIZE = 0
#
train_per_predict = TRAIN_PER_COMMON
# optimizer
OPTIMIZER_SCALE1 = Adam(lr=LR1_COMMON, decay=1e-4)
OPTIMIZER_SCALE2 = Adam(lr=LR2_COMMON, decay=1e-4)
#
threshold1 = 1.25
threshold2 = 1.25 ** 2
threshold3 = 1.25 ** 3


# option
parser = argparse.ArgumentParser(description='argparse sample.')
parser.add_argument('-s1', '--scale1', type=str, help='scale1 weights name.')
parser.add_argument('-s2', '--scale2', type=str, help='scale2 weights name.')
parser.add_argument('-se', '--seg', type=str, help='seg weights name.')
parser.add_argument('-b', '--batch', type=int, help='batch size.')
parser.add_argument('-l', '--limit', type=int, help='number of image limit size.')
parser.add_argument('-ea', '--early', type=int, help='number of early stopping.')
parser.add_argument('-ep', '--epoch', type=int, help='number of epoch.')
parser.add_argument('-lo', '--loss', type=int, help='number of select loss.')
args = parser.parse_args()
# scale1 weight path
if args.scale1:
    scale1_base_weights_path = path_the_directory + 'weights/' + args.scale1
    print("scale1 weights path\t{}".format(scale1_base_weights_path))
else:
    scale1_base_weights_path = path_the_directory + 'weights/' + \
                               'save_ex1_plane_scale1'
# seg weight path
if args.seg:
    seg_base_weights_path = path_the_directory + 'weights/' + args.scale2
    print("seg weights path\t{}".format(seg_base_weights_path))
else:
    seg_base_weights_path = None
# scale2 weight path
if args.scale2:
    scale2_base_weights_path = path_the_directory + 'weights/' + args.scale2
    print("scale2 weights path\t{}".format(scale2_base_weights_path))
else:
    scale2_base_weights_path = path_the_directory + 'weights/' + \
                               'save_ex1_plane_scale2_scale2'
# limit
if args.limit:
    LIMIT_COUNT = args.limit
    PARTITION_NUMBER = int(LIMIT_COUNT / 10) * 7
else:
    LIMIT_COUNT = 99999
    PARTITION_NUMBER = 1015
# select loss
if args.loss:
    SELECT_EVALUATION = args.loss
else:
    SELECT_EVALUATION = 1
#
if args.epoch:
    EPOCHS = args.epoch
    print("epoch size\t{}".format(EPOCHS))
else:
    EPOCHS = EPOCHS_COMMON
#
if args.batch:
    BATCH_SIZE = args.batch
    print("batch size\t{}".format(BATCH_SIZE))
else:
    BATCH_SIZE = BATCH_SIZE_COMMON
#
if args.early:
    EARLY_NUMBER = args.early
else:
    EARLY_NUMBER = EARLY_NUMBER_COMMON
#


class MyDataSet:
    def __init__(self):
        print("ins MyDataSet")

    def csv_load(self, file_path):#arg_size=(SCALE3_OUTPUT_HEIGHT, SCALE3_OUTPUT_WIDTH)):
        list_depth1 = []
        list_depth2 = []
        list_overlap = []
        with open(file_path, "r") as f:
            reader = csv.reader(f)
            for i_1, (img_filename, depth_filename, layer_filename) in enumerate(reader):
                # depth
                depth_png = keras.preprocessing.image.load_img(depth_filename, grayscale=True,
                                                               target_size=(SCALE1_OUTPUT_HEIGHT, SCALE1_OUTPUT_WIDTH))
                depth1 = keras.preprocessing.image.img_to_array(depth_png)
                depth_png = keras.preprocessing.image.load_img(depth_filename, grayscale=True,
                                                               target_size=(SCALE2_INPUT_HEIGHT, SCALE2_INPUT_WIDTH))
                depth2 = keras.preprocessing.image.img_to_array(depth_png)
                # overlap
                overlap_png = keras.preprocessing.image.load_img(layer_filename, grayscale=True,
                                                                 target_size=(OVERLAP_HEIGHT, OVERLAP_WIDTH))
                overlap = keras.preprocessing.image.img_to_array(overlap_png)
                tmp_list = np.sort(np.unique(overlap))
                # for i_2, get_num in enumerate(tmp_list):
                #     overlap[overlap == get_num] = i_2
                #
                list_depth1.append(depth1)
                list_depth2.append(depth2)
                if OVERLAP_NORMLIZE == 1:
                    list_overlap.append(overlap/OVERLAP_NUMBER)
                else:
                    list_overlap.append(overlap)
                # print("\r{0:d}".format(i_1), end="")
                if i_1 > LIMIT_COUNT - 2:
                    break
        print("")
        list_depth1 = np.asarray(list_depth1)
        list_depth1 = list_depth1.astype('float32') / 255.0
        list_depth2 = np.asarray(list_depth2)
        list_depth2 = list_depth2.astype('float32') / 255.0
        list_overlap = np.asarray(list_overlap)
        # list_overlap = list_overlap.astype('float32')
        # print("### Most overlap number\t{}".format(np.amax(self.list_overlap)))
        print("### depth.shape\t{}".format(list_depth1.shape))
        print("### overlap.shape\t{}".format(list_overlap.shape))
        return np.array(list_depth1), np.array(list_depth2), np.array(list_overlap)

    def coco_load(self, file_path, arg_size=(IMAGE_HEIGHT, IMAGE_WIDTH)):
        list_img = []
        list_label = []
        mask_count = 1
        for get_subset in ["train", "minival"]:
            if mask_count > LIMIT_COUNT - 1:
                break
            ins_coco = coco.CocoDataset()
            ins_coco.load_coco(file_path, get_subset, year='2018')
            ins_coco.prepare()
            get_ids = ins_coco.image_ids
            for i_1, image_id in enumerate(get_ids):
                if mask_count > LIMIT_COUNT:
                    break
                list_img.append(ins_coco.load_image(image_id))
                get_mask, get_class_ids = ins_coco.load_mask(image_id)
                label_array = np.zeros((arg_size[0], arg_size[1], 1))
                boolean_array = np.zeros((arg_size[0], arg_size[1], 1))
                for mask_num in np.arange(get_mask.shape[2]):
                    mask_array = np.array(get_mask[:, :, mask_num])
                    mask_array = np.reshape(mask_array, (arg_size[0], arg_size[1], 1))
                    tmp_array = np.logical_and(boolean_array == 0, mask_array == 1)
                    label_array += tmp_array * get_class_ids[mask_num]
                    boolean_array += tmp_array
                label_array = np.reshape(label_array, (label_array.shape[0], label_array.shape[1]))
                image_array1 = Image.fromarray(np.uint8(label_array))
                image_array2 = np.asarray(image_array1.resize((SCALE2_INPUT_WIDTH, SCALE2_INPUT_HEIGHT)))
                image_array2 = np.reshape(image_array2, (image_array2.shape[0], image_array2.shape[1], 1))
                list_label.append(image_array2)
                # list_label.append(label_array)
                print("\r{0:d}\t{0:d}".format(i_1, image_id), end="")
                mask_count += 1
        list_img = np.array(list_img) / 255.0
        list_img = list_img.astype('float32')
        list_label = np.array(list_label)
        list_label = list_label.astype('float32')
        print("")
        print("### img.shape\t{}".format(list_img.shape))
        print("### label.shape\t{}".format(list_label.shape))
        return list_img, list_label


data_gen_args = dict(
    zoom_range=[1, 1.5],
    rotation_range=5,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='constant',
    cval=0.0)


class Generator:
    def __init__(self):
        self.seed = random.randint(0, 1000)

    def create_generator(self, arg_x_train, arg_y1_train, arg_y2_train, arg_z_train, arg_w_train,
                         arg_x_test, arg_y1_test, arg_y2_test, arg_z_test, arg_w_test):
        # train #########
        x_train_generator = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
        y1_train_generator = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
        y2_train_generator = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
        z_train_generator = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
        w_train_generator = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
        #
        x_train_generator.fit(arg_x_train, augment=True, seed=self.seed)
        y1_train_generator.fit(arg_y1_train, augment=True, seed=self.seed)
        y2_train_generator.fit(arg_y2_train, augment=True, seed=self.seed)
        z_train_generator.fit(arg_z_train, augment=True, seed=self.seed)
        w_train_generator.fit(arg_w_train, augment=True, seed=self.seed)
        #
        x_train_iterator = x_train_generator.flow(arg_x_train, batch_size=BATCH_SIZE, seed=self.seed)
        y1_train_iterator = y1_train_generator.flow(arg_y1_train, batch_size=BATCH_SIZE, seed=self.seed)
        y2_train_iterator = y2_train_generator.flow(arg_y2_train, batch_size=BATCH_SIZE, seed=self.seed)
        z_train_iterator = z_train_generator.flow(arg_z_train, batch_size=BATCH_SIZE, seed=self.seed)
        w_train_iterator = w_train_generator.flow(arg_w_train, batch_size=BATCH_SIZE, seed=self.seed)
        #
        train_generator = (x_train_iterator, y1_train_iterator, y2_train_iterator, z_train_iterator, w_train_iterator)
        # valid ########
        x_test_generator = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
        y1_test_generator = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
        y2_test_generator = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
        z_test_generator = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
        w_test_generator = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
        #
        x_test_generator.fit(arg_x_test, augment=True, seed=self.seed)
        y1_test_generator.fit(arg_y1_test, augment=True, seed=self.seed)
        y2_test_generator.fit(arg_y2_test, augment=True, seed=self.seed)
        z_test_generator.fit(arg_z_test, augment=True, seed=self.seed)
        w_test_generator.fit(arg_w_test, augment=True, seed=self.seed)
        #
        x_test_iterator = x_test_generator.flow(arg_x_test, batch_size=BATCH_SIZE, seed=self.seed)
        y1_test_iterator = y1_test_generator.flow(arg_y1_test, batch_size=BATCH_SIZE, seed=self.seed)
        y2_test_iterator = y2_test_generator.flow(arg_y2_test, batch_size=BATCH_SIZE, seed=self.seed)
        z_test_iterator = z_test_generator.flow(arg_z_test, batch_size=BATCH_SIZE, seed=self.seed)
        w_test_iterator = w_test_generator.flow(arg_w_test, batch_size=BATCH_SIZE, seed=self.seed)
        #
        valid_generator = (x_test_iterator, y1_test_iterator, y2_test_iterator, z_test_iterator, w_test_iterator)
        return train_generator, valid_generator


class Model:
    def __init__(self):
        print('ins Model.')

    # momodel
    def get_scale1_on_RGB_net_model(self):
        input_tensor1 = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), name='image_input1')
        #
        x = Conv2D(filters=96, kernel_size=(11, 11), strides=4,
                   name='image_conv1', kernel_initializer='he_normal')(input_tensor1)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=2, name='image_pool1')(x)
        x = Conv2D(filters=256, kernel_size=(5, 5),
                   name='image_conv2', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=2, name='image_pool2')(x)
        x = Conv2D(filters=384, kernel_size=(3, 3),
                   name='conv3', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=384, kernel_size=(3, 3),
                   name='conv4', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=384, kernel_size=(3, 3),
                   name='conv5', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Flatten(name='flatten1')(x)
        x = Dense(1024, name='dence1', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(KEEP_CON)(x)
        x = Dense(TARGET_HEIGHT * TARGET_WIDTH,
                  name='dence2', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Reshape((TARGET_HEIGHT, TARGET_WIDTH, 1))(x)
        #
        scale1_prediction = x
        scale1_net_model = keras.models.Model(inputs=input_tensor1,
                                              outputs=scale1_prediction,
                                              name="scale1_net")
        return scale1_net_model

    def get_scale2_on_RGB_net_model(self):
        input_tensor1 = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), name='image_input1')
        input_tensor2 = Input(shape=(TARGET_HEIGHT, TARGET_WIDTH, 1), name='predict_input1')
        #
        #
        x = Conv2D(filters=63, kernel_size=(9, 9), strides=2,
                   name='image_conv1', kernel_initializer='he_normal')(input_tensor1)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=2, name='image_pool1')(x)
        x = Concatenate(axis=-1)([x, input_tensor2])
        insep_1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same',
                         name='insep1_conv1', kernel_initializer='he_normal')(x)
        insep_1 = BatchNormalization()(insep_1)
        insep_1 = Activation('relu')(insep_1)
        x = Conv2D(filters=64, kernel_size=(5, 5), padding='same',
                   name='conv2', kernel_initializer='he_normal')(insep_1)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=1, kernel_size=(5, 5), padding='same',
                   name='conv3', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        #
        scale2_prediction = x
        scale2_net_model = keras.models.Model(inputs=[input_tensor1, input_tensor2],
                                              outputs=scale2_prediction,
                                              name="scale2_net")
        return scale2_net_model


def scale1_invariant_error(y_true, y_pred):
    #
    first_error = K.mean(K.square(y_pred - y_true), axis=-1)
    second_error = K.square(K.sum(y_pred - y_true, axis=-1))
    second_error = second_error / K.square(K.cast_to_floatx(SCALE1_OUTPUT_HEIGHT * SCALE1_OUTPUT_WIDTH))
    error1 = first_error - 0.5 * second_error
    #
    Y = y_pred - y_true
    Y = K.batch_flatten(Y)
    Y = K.reshape(Y, (-1,) + (1, SCALE1_OUTPUT_HEIGHT, SCALE1_OUTPUT_WIDTH))
    first_error = K.pow(Y[:, :, 1:, :] - Y[:, :, :-1, :], 2)
    second_error = K.pow(Y[:, :, :, :-1] - Y[:, :, :, 1:], 2)
    error2 = K.mean(first_error) + K.mean(second_error)
    #
    return error1 + error2


def scale2_invariant_error(y_true, y_pred):
    #
    first_error = K.mean(K.square(y_pred - y_true), axis=-1)
    second_error = K.square(K.sum(y_pred - y_true, axis=-1))
    second_error = second_error / K.square(K.cast_to_floatx(SCALE2_INPUT_HEIGHT * SCALE2_INPUT_WIDTH))
    error1 = first_error - 0.5 * second_error
    #
    Y = y_pred - y_true
    Y = K.batch_flatten(Y)
    Y = K.reshape(Y, (-1,) + (1, SCALE2_INPUT_HEIGHT, SCALE2_INPUT_WIDTH))
    first_error = K.pow(Y[:, :, 1:, :] - Y[:, :, :-1, :], 2)
    second_error = K.pow(Y[:, :, :, :-1] - Y[:, :, :, 1:], 2)
    error2 = K.mean(first_error) + K.mean(second_error)
    #
    return error1 + error2


def root_mean_squared_error_linear(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def root_mean_squared_error_log(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), None))
    second_log = K.log(K.clip(y_true, K.epsilon(), None))
    return K.sqrt(K.mean(K.square(first_log - second_log), axis=-1))


def thresholded_accuracy1(y_true, y_pred):
    return K.mean(K.less(K.maximum(y_true / y_pred, y_pred / y_true), threshold1), axis=-1)


def thresholded_accuracy2(y_true, y_pred):
    return K.mean(K.less(K.maximum(y_true / y_pred, y_pred / y_true), threshold2), axis=-1)


def thresholded_accuracy3(y_true, y_pred):
    return K.mean(K.less(K.maximum(y_true / y_pred, y_pred / y_true), threshold3), axis=-1)


if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as session:
        # load data set
        ins_data_set = MyDataSet()
        depths1, depths2, overlaps = ins_data_set.csv_load(file_path=GET_TRAIN_FILE_PATH)
        images, labels = ins_data_set.coco_load(file_path=COCO_DIR)
        # depths1, depths2, overlaps, images, labels = \
        #     np.array(depths1), np.array(depths2), np.array(overlaps), np.array(images), np.array(labels)
        print("### depths{}\toverlaps{}\n### images{}\tlabels{}".format(
            depths1.shape, overlaps.shape, images.shape, labels.shape
        ))
        # instance
        ins_model = Model()
        ins_generator = Generator()
        ins_push_slack = op_push_slack.PushSlack(CHANNEL_URL1, USER_NAME)
        error_strings = ["epoch",
                         "sc-inv.",
                         "abs rel", "sqr rel",
                         "RMS(lin)", "RMS(log)",
                         "δ < 1.25**1", "δ < 1.25**2", "δ < 1.25**3",
                         "val_sc-inv.",
                         "val_abs rel", "val_sqr rel",
                         "val_RMS(lin)", "val_RMS(log)",
                         "val_δ < 1.25**1", "val_δ < 1.25**2", "val_δ < 1.25**3"
                         ]
        train_string_pos = int(len(error_strings) / 2 + 0.5)
        test_string_pos = len(error_strings)
        history_all = np.zeros((SCALE_ALL, EPOCHS + 1, test_string_pos))
        ### sstage
        data_idx = np.arange(images.shape[0])
        count_train = np.zeros((SCALE_ALL, EPOCHS + 1, test_string_pos))
        ### set model
        # scale1
        scale1_model = ins_model.get_scale1_on_RGB_net_model()
        scale1_model.load_weights(scale1_base_weights_path)
        print("scale1 load weights.")
        scale1_model.compile(optimizer=OPTIMIZER_SCALE1,
                             loss=scale1_invariant_error,
                             metrics=['mean_absolute_error', 'mean_squared_error',
                                      root_mean_squared_error_linear, root_mean_squared_error_log,
                                      thresholded_accuracy1,
                                      thresholded_accuracy2,
                                      thresholded_accuracy3,
                                      'accuracy'])
        # seg
        # SegNet_model = ins_model.get_SegNet_model()
        # SegNet_model.load_weights(seg_base_weights_path)
        # print("seg load weights.")
        # SegNet_model.compile(optimizer='adadelta',
        #                      loss="categorical_crossentropy",
        #                      metrics=['mean_absolute_error', 'mean_squared_error',
        #                               root_mean_squared_error_linear, root_mean_squared_error_log,
        #                               thresholded_accuracy1,
        #                               thresholded_accuracy2,
        #                               thresholded_accuracy3,
        #                               'accuracy'])
        # scale2
        scale2_model = ins_model.get_scale2_on_RGB_net_model()
        scale2_model.load_weights(scale2_base_weights_path)
        print("scale2 load weights.")
        scale2_model.compile(optimizer=OPTIMIZER_SCALE2,
                         loss=scale2_invariant_error,
                         metrics=['mean_absolute_error', 'mean_squared_error',
                                  root_mean_squared_error_linear, root_mean_squared_error_log,
                                  thresholded_accuracy1,
                                  thresholded_accuracy2,
                                  thresholded_accuracy3])
        ###
        class40 = np.array(
            ['BG',
             'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa',
             'table', 'door', 'window', 'bookshelf', 'picture',
             'counter', 'blinds', 'desk', 'shelves', 'curtain',
             'dresser', 'pillow', 'mirror', 'floor mat', 'clothes',
             'ceiling', 'books', 'refridgerator', 'television',
             'paper', 'towel', 'shower curtain', 'box', 'whiteboard',
             'person', 'night stand', 'toilet', 'sink', 'lamp', 'bathtub','bag',
             'otherstructure', 'otherfurniture','otherprop'])
        #
        ins_predict_to_html = op_img_to_html.PredictToHTML(HTML_PATH_BASE + PROGRAM_NAME +
                                                           "_data" + ".html")
        data_list1 = np.zeros((3, LABEL_NUMBER, images.shape[0]))
        data_list2 = np.zeros((3, 10, images.shape[0]))
        data_all1 = np.zeros((2*3, LABEL_NUMBER))
        data_all2 = np.zeros((2*3, 10))
        for for_img in range(images.shape[0]):
                # set_overlaps = np.reshape(np_utils.to_categorical(overlaps[for_img:for_img + 1], OVERLAP_NUMBER),
                #                         (overlaps[for_img:for_img + 1].shape[0], OVERLAP_HEIGHT * OVERLAP_WIDTH, OVERLAP_NUMBER))
                # set_overlaps = np_utils.to_categorical(overlaps[for_img:for_img + 1], OVERLAP_NUMBER)
                predict_depth1 = scale1_model.predict(images[for_img:for_img + 1])
                # predict_overlap = SegNet_model.predict([
                #     images[for_img:for_img + 1],
                #     predict_depth1
                # ])
                # predict_overlap = np.reshape(predict_overlap, (
                #     predict_overlap.shape[0], SegNet_OUTPUT_HEIGHT, SegNet_OUTPUT_WIDTH, OVERLAP_NUMBER))
                # predict_overlap = np.amax(predict_overlap, axis=-1)
                # predict_overlap = np.reshape(predict_overlap,
                #                              (predict_overlap.shape[0], SegNet_OUTPUT_HEIGHT, SegNet_OUTPUT_WIDTH,
                #                               1))
                predict_depth2 = scale2_model.predict([
                    images[for_img:for_img + 1],
                    predict_depth1
                ])
                predict_loss = scale2_model.test_on_batch([
                    images[for_img:for_img + 1],
                    predict_depth1
                ],
                depths2[for_img:for_img + 1])
                # label
                for get_label in range(LABEL_NUMBER):
                    label_pos = labels[for_img] == get_label
                    if np.sum(label_pos) == 0:
                        continue
                    mean_depth = np.mean(depths2[for_img][label_pos])
                    mean_predict = np.mean(predict_depth2[0][label_pos])
                    mean_error = np.mean(np.abs(depths2[for_img][label_pos] - predict_depth2[0][label_pos]))
                    data_list1[0, get_label, for_img] = mean_depth
                    data_list1[1, get_label, for_img] = mean_predict
                    data_list1[2, get_label, for_img] = mean_error
                for get_rgb in range(10):
                    rgb_pos = (depths2[for_img] >= get_rgb / 10) & (depths2[for_img] < get_rgb / 10 + 0.1)
                    if np.sum(rgb_pos) == 0:
                        continue
                    mean_depth = np.mean(depths2[for_img][rgb_pos])
                    mean_predict = np.mean(predict_depth2[0][rgb_pos])
                    mean_error = np.mean(np.abs(depths2[for_img][rgb_pos] - predict_depth2[0][rgb_pos]))
                    data_list2[0, get_rgb, for_img] = mean_depth
                    data_list2[1, get_rgb, for_img] = mean_predict
                    data_list2[2, get_rgb, for_img] = mean_error
                # save image to html
                ins_predict_to_html.convert_on_RGB_label_overlap_predict2 \
                    (images[for_img], labels[for_img], overlaps[for_img], predict_depth2[0], depths2[for_img],
                     "sc_inv " + str(predict_loss[0]) +
                     ",\tabs rel " + str(predict_loss[1]) +
                     ",\tsqr rel " + str(predict_loss[2]) +
                     ",\tRMS(lin) " + str(predict_loss[3]) +
                     ",\tRMS(log) " + str(predict_loss[4]) +
                     ",\tδ < 1.25**1 " + str(predict_loss[5]) +
                     ",\tδ < 1.25**2 " + str(predict_loss[6]) +
                     ",\tδ < 1.25**3 " + str(predict_loss[7]),
                     str("image_id {}".format(for_img)))
                print("\r{}".format(for_img), end="")
        for get_label in range(LABEL_NUMBER):
            select_pos = data_list1[0, get_label, :] != 0
            if np.sum(select_pos) == 0:
                continue
            mean_depth = np.mean(data_list1[0, get_label, :][select_pos])
            var_depth = np.var(data_list1[0, get_label, :][select_pos], ddof=1)
            data_all1[0, get_label] = mean_depth
            data_all1[1, get_label] = var_depth
            mean_depth = np.mean(data_list1[1, get_label, :][select_pos])
            var_depth = np.var(data_list1[1, get_label, :][select_pos], ddof=1)
            data_all1[2, get_label] = mean_depth
            data_all1[3, get_label] = var_depth
            mean_depth = np.mean(data_list1[2, get_label, :][select_pos])
            var_depth = np.var(data_list1[2, get_label, :][select_pos], ddof=1)
            data_all1[4, get_label] = mean_depth
            data_all1[5, get_label] = var_depth
        for get_rgb in range(10):
            select_pos = data_list2[0, get_rgb, :] != 0
            if np.sum(select_pos) == 0:
                continue
            mean_depth = np.mean(data_list2[0, get_rgb, :][select_pos])
            var_depth = np.var(data_list2[0, get_rgb, :][select_pos], ddof=1)
            data_all2[0, get_rgb] = mean_depth
            data_all2[1, get_rgb] = var_depth
            mean_depth = np.mean(data_list2[1, get_rgb, :][select_pos])
            var_depth = np.var(data_list2[1, get_rgb, :][select_pos], ddof=1)
            data_all2[2, get_rgb] = mean_depth
            data_all2[3, get_rgb] = var_depth
            mean_depth = np.mean(data_list2[2, get_rgb, :][select_pos])
            var_depth = np.var(data_list2[2, get_rgb, :][select_pos], ddof=1)
            data_all2[4, get_rgb] = mean_depth
            data_all2[5, get_rgb] = var_depth
        with open(path_the_directory + 'history/' +
                  os.path.basename(__file__) + "_label" +
                  '_history.csv', 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(class40)
            writer.writerows(data_all1)
        with open(path_the_directory + 'history/' +
                  os.path.basename(__file__) + "_depth" +
                  '_history.csv', 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(['0', '1', '2', '3', '4',
                             '5', '6', '7', '8', '9'])
            writer.writerows(data_all2)

