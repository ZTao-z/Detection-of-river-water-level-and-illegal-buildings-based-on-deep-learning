from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data

from data import BaseTransform
from data.custom_for_visual import CUSTOM_CLASSES_BUILDING as labelmap_building
from data.custom_for_visual import customDetection, customAnnotationTransform, CUSTOM_ROOT, CUSTOM_CLASSES_BUILDING

# from ssd import build_ssd
from ssd_resnet_101 import build_ssd

import sys
import os
import time
import argparse
import numpy as np
import pickle
import cv2
import math

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model_building',
                    default='useful_weight/CUSTOM.pth', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.15, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--custom_root', default=CUSTOM_ROOT, help='Location of VOC root directory')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def test_net(save_folder, net, cuda, testset, transform, thresh, labelmap):
    # dump predictions and assoc. ground truth to text file for now
    filename = save_folder + 'result_%s.txt'
    num_images = len(testset)
    for i in range(num_images):
        print('Testing image {:d}/{:d}....'.format(i+1, num_images))
        img = testset.pull_image(i)
        img_id, annotation = testset.pull_anno(i)
        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))

        if cuda:
            x = x.cuda()

        y = net(x)      # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])
        pred_num = 0
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= args.visual_threshold:
                score = detections[0, i, j, 0]
                label_name = labelmap[i-1]
                pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
                coords = (pt[0], pt[1], pt[2], pt[3])
                pred_num += 1
                with open(filename % label_name, mode='a') as f:
                    f.write(str(img_id) + ' ' +
                            str(score.cpu().numpy()) + ' '+ ' '.join(str(c) for c in coords) + '\n')
                j += 1

def xmlData(name, width, height, label):
    return '''<annotation>
    <folder>JPEGImages</folder>
    <filename>%s.jpg</filename>
    <path>%s.jpg</path>
    <source>
        <database>Unknown</database>
    </source>
    <size>
        <width>%d</width>
        <height>%d</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
    <object>
        <name>%s</name>
        <pose>Unspecified</pose>
        <truncated>1</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>0</xmin>
            <ymin>0</ymin>
            <xmax>1</xmax>
            <ymax>1</ymax>
        </bndbox>
    </object>
</annotation>''' % (name, name, width, height, label)

def get_output_dir(name, phase=""):
    filedir = os.path.join(name, phase)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir

def is_rect_intersect(rect1, rect2):
    rect1_x1 = math.floor(rect1['x1'])
    rect1_y1 = math.floor(rect1['y1'])
    rect1_x2 = math.floor(rect1['x2'])
    rect1_y2 = math.floor(rect1['y2'])

    rect2_x1 = math.floor(rect2['x1'])
    rect2_y1 = math.floor(rect2['y1'])
    rect2_x2 = math.floor(rect2['x2'])
    rect2_y2 = math.floor(rect2['y2'])

    zx = abs(rect1_x1 + rect1_x2 - rect2_x1 - rect2_x2)
    x = abs(rect1_x1 - rect1_x2) + abs(rect2_x1 - rect2_x2)

    zy = abs(rect1_y1 + rect1_y2 - rect2_y1 - rect2_y2)
    y = abs(rect1_y1 - rect1_y2) + abs(rect2_y1 - rect2_y2)

    return True if zx <= x and zy <= y else False


def test_custom():
    DEBUG = False
    set_type = 'test'

    if not os.path.exists(os.path.join(args.save_folder, 'result_building.txt')):
        # load net
        num_classes_building = len(labelmap_building) + 1                      # +1 for background
        net = build_ssd('test', 300, num_classes_building)            # initialize SSD
        net.load_state_dict(torch.load(args.trained_model_building))
        net.eval()

        print('Finished loading model!')
        # load data
        dataset1 = customDetection(args.custom_root, [('buildingwater', set_type)], None, customAnnotationTransform(class_to_ind=dict(zip(CUSTOM_CLASSES_BUILDING, range(len(CUSTOM_CLASSES_BUILDING))))))
        if args.cuda:
            net = net.cuda()
            cudnn.benchmark = True
        # evaluation
    
        test_net(args.save_folder, net, args.cuda, dataset1,
                BaseTransform(net.size, (104, 117, 123)),
                thresh=args.visual_threshold, labelmap=labelmap_building)

    rootPath = 'F:/ssd/data/video/buildingwater'
    img_path = os.path.join(rootPath, 'JPEGImages', '%s.jpg')
    imgList_building = {}
    imgList_water = {}

    with open(os.path.join(args.save_folder, 'result_building.txt'), 'r') as f:
        text_lines = f.readlines()
        for line in text_lines:
            info = line.split(" ")
            name, score, x1, y1, x2, y2 = info
            if name in imgList_building:
                imgList_building[name].append({
                    'score': float(score),
                    'x1': float(x1),
                    'y1': float(y1),
                    'x2': float(x2),
                    'y2': float(y2)
                })
            else:
                imgList_building[name] = [{
                    'score': float(score),
                    'x1': float(x1),
                    'y1': float(y1),
                    'x2': float(x2),
                    'y2': float(y2)
                }]
    
    with open(os.path.join(args.save_folder, 'result_water.txt'), 'r') as f:
        text_lines = f.readlines()
        for line in text_lines:
            info = line.split(" ")
            name, score, x1, y1, x2, y2 = info
            if name in imgList_water:
                imgList_water[name].append({
                    'score': float(score),
                    'x1': float(x1),
                    'y1': float(y1),
                    'x2': float(x2),
                    'y2': float(y2)
                })
            else:
                imgList_water[name] = [{
                    'score': float(score),
                    'x1': float(x1),
                    'y1': float(y1),
                    'x2': float(x2),
                    'y2': float(y2)
                }]

    opacity = 0.8
    for name in imgList_building:
        img_building = imgList_building[name]
        img_water = imgList_water[name] if name in imgList_water else []
        
        image = cv2.imread(img_path % name)
        (h, w, c) = image.shape
        img_black = image.copy()
        img_cp = image.copy()
        img_black.fill(1)


        for building in img_building:
            for water in img_water:
                if is_rect_intersect(building, water):
                    x1_b = max(math.floor(building['x1']), 0)
                    y1_b = max(math.floor(building['y1']), 0)
                    x2_b = min(math.floor(building['x2']), w)
                    y2_b = min(math.floor(building['y2']), h)
                    cv2.rectangle(image, (x1_b-2, y1_b-2), (x2_b+2, y2_b+2), (0,0,255), 5)
                    img_black[y1_b:y2_b, x1_b:x2_b] = 0
        

        # for building in img_building:
        #     x1_b = max(math.floor(building['x1']), 0)
        #     y1_b = max(math.floor(building['y1']), 0)
        #     x2_b = min(math.floor(building['x2']), w)
        #     y2_b = min(math.floor(building['y2']), h)
        #     # cv2.rectangle(image, (x1_b, y1_b), (x2_b, y2_b), (0,0,255), 5)
        #     img_black[y1_b:y2_b, x1_b:x2_b] = 0
        image[:,:,0] = (1 - img_black[:,:,0]) * (img_cp[:,:,0]) + img_black[:,:,0] * image[:,:,0]
        image[:,:,1] = (1 - img_black[:,:,1]) * (img_cp[:,:,1]) + img_black[:,:,1] * image[:,:,1]
        image[:,:,2] = (1 - img_black[:,:,2]) * (img_cp[:,:,2] ) + img_black[:,:,2] * image[:,:,2]

        image[:,:,0] = (1 - img_black[:,:,0]) * (image[:,:,0] * opacity + 0 * (1 - opacity)) + img_black[:,:,0] * image[:,:,0]
        image[:,:,1] = (1 - img_black[:,:,1]) * (image[:,:,1] * opacity + 0 * (1 - opacity)) + img_black[:,:,1] * image[:,:,1]
        image[:,:,2] = (1 - img_black[:,:,2]) * (image[:,:,2] * opacity + 255 * (1 - opacity)) + img_black[:,:,2] * image[:,:,2]

        # for water in img_water:
        #     x1_w = max(math.floor(water['x1']), 0)
        #     y1_w = max(math.floor(water['y1']), 0)
        #     x2_w = min(math.floor(water['x2']), w)
        #     y2_w = min(math.floor(water['y2']), h)
        #     cv2.rectangle(image, (x1_w, y1_w), (x2_w, y2_w), (0,255,0), 5)
        
        image = cv2.resize(image, (512, 512))
        # cv2.putText(image, 'building', (10, 40), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 255), 2)
        # cv2.putText(image, 'water', (10, 80), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 255, 0), 2)
        cv2.imshow('w2', image)
        cv2.waitKey()

if __name__ == '__main__':
    test_custom()
