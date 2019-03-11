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
TRAIN_FILE_PATH_SELECT = 'imageDepthOverlap_partial_3amax5.csv'
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
# scale2 weight path
if args.scale2:
    scale2_base_weights_path = path_the_directory + 'weights/' + args.scale2
    print("scale2 weights path\t{}".format(scale2_base_weights_path))
else:
    scale2_base_weights_path = None
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
    SELECT_EVALUATION = 8
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
                list_label.append(label_array)
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

    def get_scale2_on_RGB_overlap_net_model(self):
        input_tensor1 = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), name='image_input1')
        input_tensor2 = Input(shape=(TARGET_HEIGHT, TARGET_WIDTH, 1), name='predict_input1')
        input_tensor3 = Input(shape=(TARGET_HEIGHT, TARGET_WIDTH, 1), name='overlap_input1')
        #
        x = Conv2D(filters=63, kernel_size=(9, 9), strides=2,
                   name='image_conv1', kernel_initializer='he_normal')(input_tensor1)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=2, name='image_pool1')(x)
        x = Concatenate(axis=-1)([x, input_tensor2, input_tensor3])
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
        scale2_net_model = keras.models.Model(inputs=[input_tensor1, input_tensor2, input_tensor3],
                                              outputs=scale2_prediction,
                                              name="scale2_net")
        return scale2_net_model

    def get_scale2_on_RGB_one_hot_overlap_net_model(self):
        input_tensor1 = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), name='image_input1')
        input_tensor2 = Input(shape=(TARGET_HEIGHT, TARGET_WIDTH, 1), name='predict_input1')
        input_tensor3 = Input(shape=(OVERLAP_HEIGHT, OVERLAP_WIDTH, OVERLAP_NUMBER), name='overlap_input1')
        #
        x = Conv2D(filters=63, kernel_size=(9, 9), strides=2,
                   name='image_conv1', kernel_initializer='he_normal')(input_tensor1)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=2, name='image_pool1')(x)
        x = Concatenate(axis=-1)([x, input_tensor2, input_tensor3])
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
        scale2_net_model = keras.models.Model(inputs=[input_tensor1, input_tensor2, input_tensor3],
                                              outputs=scale2_prediction,
                                              name="scale2_net")
        return scale2_net_model

    def get_SegNet_model(self, classes=OVERLAP_NUMBER):
        ### @ https://github.com/alexgkendall/SegNet-Tutorial/blob/master/Example_Models/bayesian_segnet_camvid.prototxt
        input_tensor1 = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), name='image_input1')
        input_tensor2 = Input(shape=(SCALE1_OUTPUT_HEIGHT, SCALE1_OUTPUT_WIDTH, 1), name='predict_input1')
        #
        y = UpSampling2D((4, 4), name='upsampling1')(input_tensor2)
        y = ZeroPadding2D(padding=((4, 4), (4, 4)))(y)
        x = Concatenate(axis=-1)([input_tensor1, y])
        #
        insep_1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same',
                         name='insep1_conv1', kernel_initializer='he_normal')(x)
        insep_1 = BatchNormalization()(insep_1)
        insep_1 = Activation('relu')(insep_1)
        # Encoder
        x = Conv2D(32, (3, 3), padding="same", kernel_initializer='he_normal')(insep_1)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(64, (3, 3), padding="same", kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(128, (3, 3), padding="same", kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(256, (3, 3), padding="same", kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        # Decoder
        x = Conv2D(256, (3, 3), padding="same", kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(128, (3, 3), padding="same", kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(64, (3, 3), padding="same", kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(32, (3, 3), padding="same", kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = ZeroPadding2D(padding=((2, 2), (0, 0)))(x)

        x = Conv2D(filters=64, kernel_size=(9, 9), strides=2, name='reshape_conv1', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=2, name='reshape_pool1')(x)

        x = Conv2D(classes, (1, 1), padding="valid")(x)
        x = Activation("softmax")(x)
        x = Reshape((SegNet_OUTPUT_HEIGHT * SegNet_OUTPUT_WIDTH, classes))(x)
        SegNet_prediction = x
        SegNet_model = keras.models.Model(inputs=[input_tensor1, input_tensor2],
                                   outputs=SegNet_prediction,
                                   name="seg_net")
        # SegNet_model.summary()
        return SegNet_model


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
                         "categorical_crossentropy",
                         "abs rel", "sqr rel",
                         "RMS(lin)", "RMS(log)",
                         "δ < 1.25**1", "δ < 1.25**2", "δ < 1.25**3",
                         "accuracy",
                         "val_categorical_crossentropy",
                         "val_abs rel", "val_sqr rel",
                         "val_RMS(lin)", "val_RMS(log)",
                         "val_δ < 1.25**1", "val_δ < 1.25**2", "val_δ < 1.25**3",
                         "val_accuracy",
                         ]
        train_string_pos = int(len(error_strings) / 2 + 0.5)
        test_string_pos = len(error_strings)
        history_all = np.zeros((SCALE_ALL, EPOCHS + 1, test_string_pos))
        ### sstage
        data_idx = np.arange(images.shape[0])
        count_train = np.zeros((SCALE_ALL, EPOCHS + 1, test_string_pos))
        for for_stage in range(STAGE_ALL):
            np.random.shuffle(data_idx)
            x_train, x_test = np.array(images[data_idx[:PARTITION_NUMBER]]),\
                              np.array(images[data_idx[PARTITION_NUMBER:]])
            y1_train, y1_test = np.array(depths1[data_idx[:PARTITION_NUMBER]]), \
                                np.array(depths1[data_idx[PARTITION_NUMBER:]])
            y2_train, y2_test = np.array(depths2[data_idx[:PARTITION_NUMBER]]), \
                                np.array(depths2[data_idx[PARTITION_NUMBER:]])
            z_train, z_test = np.array(overlaps[data_idx[:PARTITION_NUMBER]]), \
                              np.array(overlaps[data_idx[PARTITION_NUMBER:]])
            w_train, w_test = np.array(labels[data_idx[:PARTITION_NUMBER]]), \
                              np.array(labels[data_idx[PARTITION_NUMBER:]])
            #
            train_generator, valid_generator =\
                ins_generator.create_generator(x_train, y1_train, y2_train, z_train, w_train,
                                                                              x_test, y1_test, y2_test, z_test, w_test)
            x_train_generator, y1_train_generator, y2_train_generator, z_train_generator, w_train_generator =\
                train_generator[0], train_generator[1], train_generator[2], train_generator[3], train_generator[4]
            x_test_generator, y1_test_generator, y2_test_generator, z_test_generator, w_test_generator =\
                valid_generator[0], valid_generator[1], valid_generator[2], valid_generator[3], valid_generator[4]
            #
            if SELECT_EVALUATION > 4:
                tmp_times = -1
            else:
                tmp_times = 1
            ### sscale
            for for_scale in range(SCALE_ALL):
                for_scale += 1
                # chch
                if for_scale == 1:
                    scale1_model = ins_model.get_scale1_on_RGB_net_model()
                    if scale1_base_weights_path is not None:
                        print("load weights.")
                        scale1_model.load_weights(scale1_base_weights_path)
                        scale1_model.compile(optimizer=OPTIMIZER_SCALE1,
                                             loss=scale1_invariant_error,
                                             metrics=['mean_absolute_error', 'mean_squared_error',
                                                      root_mean_squared_error_linear, root_mean_squared_error_log,
                                                      thresholded_accuracy1,
                                                      thresholded_accuracy2,
                                                      thresholded_accuracy3,
                                                      'accuracy'])
                        continue
                    scale1_model.compile(optimizer=OPTIMIZER_SCALE1,
                                         loss=scale1_invariant_error,
                                         metrics=['mean_absolute_error', 'mean_squared_error',
                                                  root_mean_squared_error_linear, root_mean_squared_error_log,
                                                  thresholded_accuracy1,
                                                  thresholded_accuracy2,
                                                  thresholded_accuracy3,
                                                  'accuracy'])
                elif for_scale == 2:
                    scale2_model = ins_model.get_SegNet_model()
                    if scale2_base_weights_path is not None:
                        print("load weights.")
                        scale2_model.load_weights(scale2_base_weights_path)
                    scale2_model.compile(optimizer='adadelta',
                                         loss="categorical_crossentropy",
                                         metrics=['mean_absolute_error', 'mean_squared_error',
                                                  root_mean_squared_error_linear, root_mean_squared_error_log,
                                                  thresholded_accuracy1,
                                                  thresholded_accuracy2,
                                                  thresholded_accuracy3,
                                                  'accuracy'])
                history1 = np.zeros((1, EPOCHS + 1, test_string_pos))
                count_train[for_scale - 1, :, 0] = \
                    count_train[for_scale - 1, :, 0] + 1
                count_train[for_scale - 1, 0, 1:] = \
                    count_train[for_scale - 1, 0, 1:] + 1
                for for_1 in range(EPOCHS + 1):
                    history1[0, for_1, 0] = for_1
                # set variable
                best_evaluation = 99999
                # train start
                ins_push_slack.send_text("〓Start\t" + os.path.abspath(__file__) +
                                         "\n〓stage" + str(for_stage) +
                                         "\n〓scale" + str(for_scale))
                # set variable
                early_stopping_flag = 0
                #
                batch_per_sample = int(PARTITION_NUMBER / BATCH_SIZE)
                if PARTITION_NUMBER % BATCH_SIZE > 0:
                    batch_per_sample += 1
                test_per_sample = int(batch_per_sample / 2.5) + 1
                ### 0epoch history
                # train
                predict_loss = np.zeros(train_string_pos - 1)
                for index_number in range(batch_per_sample):
                    get_x_train = np.array(x_train_generator.next())
                    get_y1_train = np.array(y1_train_generator.next())
                    # get_y2_train = np.array(y2_train_generator.next())
                    get_z_train = np.array(z_train_generator.next())
                    # set_z_train = np_utils.to_categorical(get_z_train, OVERLAP_NUMBER)
                    set_z_train = np.reshape(np_utils.to_categorical(get_z_train, OVERLAP_NUMBER),
                                             (get_z_train.shape[0], OVERLAP_HEIGHT * OVERLAP_WIDTH, OVERLAP_NUMBER))
                    if for_scale == 1:
                        predict_loss = predict_loss + \
                                       np.array(scale1_model.test_on_batch(
                                           get_x_train,
                                           get_y1_train))
                    elif for_scale == 2:
                        predict_depth = scale1_model.predict(get_x_train)
                        predict_loss = predict_loss + \
                                       np.array(scale2_model.test_on_batch(
                                           [
                                               get_x_train,
                                               predict_depth
                                           ],
                                           set_z_train))
                predict_loss = predict_loss / batch_per_sample
                history1[0, 0, 1:train_string_pos] = predict_loss
                # test
                predict_loss = np.zeros(train_string_pos - 1)
                for for_1 in range(test_per_sample):
                    select_sep1 = BATCH_SIZE * for_1
                    select_sep2 = BATCH_SIZE * (for_1 + 1)
                    if select_sep2 > x_test.shape[0]:
                        select_sep2 = x_test.shape[0] - 1
                    if for_scale == 1:
                        predict_loss = predict_loss + \
                                       np.array(scale1_model.test_on_batch(
                                           x_test[select_sep1:select_sep2],
                                           y1_test[select_sep1:select_sep2]))
                    elif for_scale == 2:
                        set_z_test = np.reshape(np_utils.to_categorical(z_test[select_sep1:select_sep2], OVERLAP_NUMBER),
                                                (z_test[select_sep1:select_sep2].shape[0], OVERLAP_HEIGHT * OVERLAP_WIDTH, OVERLAP_NUMBER))
                        predict_depth = scale1_model.predict(x_test[select_sep1:select_sep2])
                        predict_loss = predict_loss + \
                                       np.array(scale2_model.test_on_batch(
                                           [
                                               x_test[select_sep1:select_sep2],
                                               predict_depth
                                           ],
                                           set_z_test))
                predict_loss = predict_loss / test_per_sample
                history1[0, 0, train_string_pos:test_string_pos] = predict_loss
                ### eepoch
                for for_epoch in range(EPOCHS):
                    count_train[for_scale - 1, for_epoch + 1, 1:] = \
                        count_train[for_scale - 1, for_epoch + 1, 1:] + 1
                    for index_number in range(batch_per_sample):
                        # train batch
                        get_x_train = np.array(x_train_generator.next())
                        get_y1_train = np.array(y1_train_generator.next())
                        # get_y2_train = np.array(y2_train_generator.next())
                        get_z_train = np.array(z_train_generator.next())
                        # set_z_train = np_utils.to_categorical(get_z_train, OVERLAP_NUMBER)
                        set_z_train = np.reshape(np_utils.to_categorical(get_z_train, OVERLAP_NUMBER),
                                                 (get_z_train.shape[0], OVERLAP_HEIGHT * OVERLAP_WIDTH, OVERLAP_NUMBER))
                        # get_w_train = np.array(w_train_generator.next())
                        # set_w_train = np_utils.to_categorical(get_w_train, 41)
                        # train
                        if for_scale == 1:
                            predict_loss = scale1_model.train_on_batch(get_x_train, get_y1_train)
                        elif for_scale == 2:
                            predict_depth = scale1_model.predict(get_x_train)
                            predict_loss = scale2_model.train_on_batch([
                                get_x_train,
                                predict_depth
                            ],
                                set_z_train)
                        # print
                        get_caption = "〓stage" + str(for_stage) + "\tscale " + str(for_scale) +\
                                      "\tfor_epoch " + str(for_epoch) + \
                                      ",\tIndex_number " + str(index_number)
                        # print("\r" + get_caption)
                        sys.stdout.write("\r" + get_caption)
                        sys.stdout.flush()
                    # append history
                    predict_loss = np.zeros(train_string_pos - 1)
                    for index_number in range(train_per_predict):
                        get_x_train = np.array(x_train_generator.next())
                        get_y1_train = np.array(y1_train_generator.next())
                        # get_y2_train = np.array(y2_train_generator.next())
                        get_z_train = np.array(z_train_generator.next())
                        # set_z_train = np_utils.to_categorical(get_z_train, OVERLAP_NUMBER)
                        set_z_train = np.reshape(np_utils.to_categorical(get_z_train, OVERLAP_NUMBER),
                                                 (get_z_train.shape[0], OVERLAP_HEIGHT * OVERLAP_WIDTH, OVERLAP_NUMBER))
                        if for_scale == 1:
                            predict_loss = predict_loss + \
                                           np.array(scale1_model.test_on_batch(
                                               get_x_train,
                                               get_y1_train))
                        elif for_scale == 2:
                            predict_depth = scale1_model.predict(get_x_train)
                            predict_loss = predict_loss + \
                                           np.array(scale2_model.test_on_batch([
                                               get_x_train,
                                               predict_depth
                                           ],
                                               set_z_train))
                    predict_loss = predict_loss / train_per_predict
                    history1[0, for_epoch + 1, 1:train_string_pos] = predict_loss
                    #
                    ### test batch
                    predict_loss = np.zeros(train_string_pos - 1)
                    for for_1 in range(test_per_sample):
                        select_sep1 = BATCH_SIZE*for_1
                        select_sep2 = BATCH_SIZE*(for_1+1)
                        if select_sep2 > x_test.shape[0]:
                            select_sep2 = x_test.shape[0] - 1
                        if for_scale == 1:
                            predict_loss = predict_loss + \
                                           np.array(scale1_model.test_on_batch(
                                               x_test[select_sep1:select_sep2],
                                               y1_test[select_sep1:select_sep2]))
                        elif for_scale == 2:#this
                            set_z_test = np.reshape(np_utils.to_categorical(z_test[select_sep1:select_sep2], OVERLAP_NUMBER),
                                                    (z_test[select_sep1:select_sep2].shape[0], OVERLAP_HEIGHT * OVERLAP_WIDTH, OVERLAP_NUMBER))
                            predict_depth = scale1_model.predict(x_test[select_sep1:select_sep2])
                            predict_loss = predict_loss + \
                                           np.array(scale2_model.test_on_batch([
                                               x_test[select_sep1:select_sep2],
                                               predict_depth
                                           ],
                                               set_z_test))
                            # print("1 {}".format(predict_loss))
                    predict_loss = predict_loss / test_per_sample
                    history1[0, for_epoch + 1, train_string_pos:test_string_pos] = predict_loss
                    # predict
                    # get_rand = np.random.randint(0, x_test.shape[0])
                    # if for_scale == 1:
                    #     predict_depth = scale1_model.predict(x_test[get_rand:get_rand+1])
                    #     predict_depth = predict_depth.astype(np.float32)
                    # early stop
                    if best_evaluation > predict_loss[SELECT_EVALUATION] * tmp_times:
                        best_evaluation = predict_loss[SELECT_EVALUATION] * tmp_times
                        early_stopping_flag = 0
                        print("")
                        print("best evaluation " + str(best_evaluation * tmp_times))
                        ins_push_slack.send_text(get_caption +
                                                 "\nbest evaluation " + str(predict_loss) +
                                                 "\nloss " + str(predict_loss[0]))
                        print("loss " + str(predict_loss[0]) +
                              ",\tevaluation " + str(predict_loss[1:]))
                        # save weights
                        tmp_path1 = path_the_directory + 'weights/'
                        tmp_path2 = PROGRAM_NAME +'_scale' + str(for_scale) +\
                                    '_renewal_val_' + str(predict_loss[SELECT_EVALUATION])
                        tmp_path3 = PROGRAM_NAME
                        save_weight_path = tmp_path1 + tmp_path2
                        if for_scale == 1:
                            scale1_model.save_weights(save_weight_path)
                        if for_scale == 2:
                            scale2_model.save_weights(save_weight_path)
                    else:
                        early_stopping_flag += 1
                        # print("")
                        # print("loss " + str(predict_loss[0]) +
                        #       ",\tevaluation " + str(predict_loss[1:]))
                        if early_stopping_flag > EARLY_NUMBER:
                            break
                history_all[for_scale-1, :, :] = history_all[for_scale-1, :, :] + history1
                print("")
                ins_push_slack.send_text("〓End\t" + os.path.abspath(__file__) +
                                         "\n〓stage" + str(for_stage) +
                                         "\n〓scale" + str(for_scale))
        count_train = np.clip(count_train, 1, STAGE_ALL)
        history_all = history_all / count_train
        # history to csv
        for for_scale in range(SCALE_ALL):
            for_scale = for_scale+1
            if scale1_base_weights_path is not None:
                if for_scale == 1:
                    continue
            if for_scale == 1:
                with open(path_the_directory + 'history/' +
                          os.path.basename(__file__) + "_scale" + str(for_scale) +
                          '_history.csv', 'w') as f:
                    writer = csv.writer(f, lineterminator='\n')
                    error_string1 = error_strings
                    for for_2 in range(train_string_pos - 1):
                        for_2 += train_string_pos
                        error_string1[for_2] = 'plane_' + error_strings[for_2]
                    writer.writerow(error_string1)
                    writer.writerows(history_all[for_scale-1])
            if for_scale == 2:
                with open(path_the_directory + 'history/' +
                          os.path.basename(__file__) + "_scale" + str(for_scale) +
                          '_history.csv', 'w') as f:
                    writer = csv.writer(f, lineterminator='\n')
                    error_string1 = error_strings
                    for for_2 in range(train_string_pos - 1):
                        for_2 += train_string_pos
                        error_string1[for_2] = 'seg_partial5_' + error_strings[for_2]
                    writer.writerow(error_string1)
                    writer.writerows(history_all[for_scale-1])
            ins_predict_to_html = op_img_to_html.PredictToHTML(HTML_PATH_BASE + PROGRAM_NAME +
                                                               "_scale" + str(1) +
                                                               ".html")
            for for_img in range(x_test.shape[0]):
                if for_scale == 1:
                    predict_depth = scale1_model.predict(images[for_img:for_img + 1])
                    predict_loss = scale1_model.test_on_batch(images[for_img:for_img + 1],
                                                              depths1[for_img:for_img + 1])
                    # save image to html
                    ins_predict_to_html.convert_on_RGB_label_overlap_predict2 \
                        (x_test[for_img], w_test[for_img], z_test[for_img], predict_depth[0], y2_test[for_img],
                         "sc_inv " + str(predict_loss[0]) +
                         ",\tabs rel " + str(predict_loss[1]) +
                         ",\tsqr rel " + str(predict_loss[2]) +
                         ",\tRMS(lin) " + str(predict_loss[3]) +
                         ",\tRMS(log) " + str(predict_loss[4]) +
                         ",\tδ < 1.25**1 " + str(predict_loss[5]) +
                         ",\tδ < 1.25**2 " + str(predict_loss[6]) +
                         ",\tδ < 1.25**3 " + str(predict_loss[7]) +
                         "\taccuracy " + str(predict_loss[8]),
                         get_caption)
                if for_scale == 2:
                    set_overlaps = np.reshape(np_utils.to_categorical(overlaps[for_img:for_img + 1], OVERLAP_NUMBER),
                                            (overlaps[for_img:for_img + 1].shape[0], OVERLAP_HEIGHT * OVERLAP_WIDTH, OVERLAP_NUMBER))
                    # set_overlaps = np_utils.to_categorical(overlaps[for_img:for_img + 1], OVERLAP_NUMBER)
                    predict_depth1 = scale1_model.predict(images[for_img:for_img + 1])
                    predict_overlap = scale2_model.predict([
                        images[for_img:for_img + 1],
                        predict_depth1
                    ])
                    predict_loss = scale2_model.test_on_batch([
                        images[for_img:for_img + 1],
                        predict_depth1,
                    ],
                    set_overlaps)
                    predict_overlap = np.reshape(predict_overlap, (
                        predict_overlap.shape[0], SegNet_OUTPUT_HEIGHT, SegNet_OUTPUT_WIDTH, OVERLAP_NUMBER))
                    predict_overlap = np.amax(predict_overlap, axis=-1)
                    predict_overlap = np.reshape(predict_overlap,
                                                 (predict_overlap.shape[0], SegNet_OUTPUT_HEIGHT, SegNet_OUTPUT_WIDTH, 1))
                    # save image to html
                    ins_predict_to_html.convert_on_RGB_label_overlap_predict1 \
                        (x_test[for_img], w_test[for_img], z_test[for_img], predict_overlap[0], y2_test[for_img],
                         "categorical_crossentropy" + str(predict_loss[0]) +
                         ",\tabs rel " + str(predict_loss[1]) +
                         ",\tsqr rel " + str(predict_loss[2]) +
                         ",\tRMS(lin) " + str(predict_loss[3]) +
                         ",\tRMS(log) " + str(predict_loss[4]) +
                         ",\tδ < 1.25**1 " + str(predict_loss[5]) +
                         ",\tδ < 1.25**2 " + str(predict_loss[6]) +
                         ",\tδ < 1.25**3 " + str(predict_loss[7]) +
                         "\taccuracy " + str(predict_loss[8]),
                         get_caption)

