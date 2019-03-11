import logging
from logging import FileHandler
from vlogging import VisualRecord
from PIL import Image
import numpy as np
from time import sleep
import io
#
import matplotlib.pyplot as plt
# User function
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/.')
from Operation.common import *


class PredictToHTML():
    def __init__(self, arg_name):
        self.logger = logging.getLogger("hoge")
        self.ins_file_handler = FileHandler(arg_name, mode="w")
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(self.ins_file_handler)
        self.img_image = None
        self.depth_image = None
        self.label_image = None
        self.overlap_image = None
        self.predict_image = None

    def convert_on_RGB_label_overlap_predict1(self, arg_img, arg_label, arg_overlap, arg_predict, arg_depth, arg_loss, arg_caption):
        # arg_img = np.array(arg_img)
        # arg_label = np.array(arg_label)
        # arg_predict = np.array(arg_predict)
        # arg_depth = np.array(arg_depth)
        #
        cmap = plt.get_cmap('jet')
        # img
        arg_img = Image.fromarray(np.uint8(arg_img * 255))
        self.img_image = arg_img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        # label
        arg_label = np.delete(cmap(np.reshape(arg_label / np.amax(arg_label), (arg_label.shape[:-1]))), 3, 2) * 255
        label_array = Image.fromarray(np.uint8(arg_label))
        self.label_image = label_array.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        # overlap
        arg_overlap = np.delete(cmap(np.reshape(arg_overlap / np.amax(arg_overlap), (arg_overlap.shape[:-1]))), 3,
                                2) * 255
        overlap_array = Image.fromarray(np.uint8(arg_overlap))
        self.overlap_image = overlap_array.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        # predict
        arg_predict = np.delete(cmap(np.reshape(arg_predict / np.amax(arg_predict), (arg_predict.shape[:-1]))), 3, 2) * 255
        predict_array = Image.fromarray(np.uint8(arg_predict))
        self.predict_image = predict_array.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        # depth
        arg_depth = np.delete(cmap(np.reshape(np.clip(arg_depth, 0, 1), (arg_depth.shape[:-1]))), 3, 2) * 255
        depth_array = Image.fromarray(np.uint8(arg_depth))
        self.depth_image = depth_array.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        #
        self.logger.debug(VisualRecord(
            arg_loss, [self.img_image, self.label_image, self.overlap_image, self.predict_image, self.depth_image], arg_caption,
            fmt="jpeg"
        ))

    # predict/np.amax(predict) off
    def convert_on_RGB_label_overlap_predict2(self, arg_img, arg_label, arg_overlap, arg_predict, arg_depth, arg_loss, arg_caption):
        # arg_img = np.array(arg_img)
        # arg_label = np.array(arg_label)
        # arg_predict = np.array(arg_predict)
        # arg_depth = np.array(arg_depth)
        #
        cmap = plt.get_cmap('jet')
        # img
        arg_img = Image.fromarray(np.uint8(arg_img * 255))
        self.img_image = arg_img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        # label
        arg_label = np.delete(cmap(np.reshape(arg_label / np.amax(arg_label), (arg_label.shape[:-1]))), 3, 2) * 255
        label_array = Image.fromarray(np.uint8(arg_label))
        self.label_image = label_array.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        # overlap
        arg_overlap = np.delete(cmap(np.reshape(arg_overlap / np.amax(arg_overlap), (arg_overlap.shape[:-1]))), 3,
                                2) * 255
        overlap_array = Image.fromarray(np.uint8(arg_overlap))
        self.overlap_image = overlap_array.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        # predict
        arg_predict = np.delete(cmap(np.reshape(np.clip(arg_predict, 0, 1), (arg_predict.shape[:-1]))), 3, 2) * 255
        predict_array = Image.fromarray(np.uint8(arg_predict))
        self.predict_image = predict_array.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        # depth
        arg_depth = np.delete(cmap(np.reshape(np.clip(arg_depth, 0, 1), (arg_depth.shape[:-1]))), 3, 2) * 255
        depth_array = Image.fromarray(np.uint8(arg_depth))
        self.depth_image = depth_array.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        #
        self.logger.debug(VisualRecord(
            arg_loss, [self.img_image, self.label_image, self.overlap_image, self.predict_image, self.depth_image], arg_caption,
            fmt="jpeg"
        ))
