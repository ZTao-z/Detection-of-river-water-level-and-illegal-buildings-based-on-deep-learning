from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data

from data import BaseTransform
from data.custom_for_visual import CUSTOM_CLASSES_GAUGE as labelmap_gauge
from data.custom_for_visual import CUSTOM_CLASSES_WATERLINE as labelmap_waterline
from data.custom_for_visual import customDetection, customAnnotationTransform, CUSTOM_ROOT, CUSTOM_CLASSES_GAUGE, CUSTOM_CLASSES_WATERLINE

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
parser.add_argument('--trained_model_gauge',
                    default='useful_weight/CUSTOM_gauge.pth', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--trained_model_waterline',
                    default='useful_weight/CUSTOM_mark.pth', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.1, type=float,
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
            while detections[0, i, j, 0] >= 0.1:
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

def test_custom():
    DEBUG = False
    set_type = 'test'
    devkit_path = args.custom_root + 'test'
    devkit_annopath = os.path.join(args.custom_root, 'test', 'Annotations')
    devkit_imgpath = os.path.join(args.custom_root, 'test', 'JPEGImages')
    devkit_imgsetpath = os.path.join(args.custom_root, 'test', 'ImageSets', 'Main')

    # load net
    num_classes_gauge = len(labelmap_gauge) + 1                      # +1 for background
    net = build_ssd('test', 300, num_classes_gauge)            # initialize SSD
    net.load_state_dict(torch.load(args.trained_model_gauge))
    net.eval()

    num_classes_waterline = len(labelmap_waterline) + 1                      # +1 for background
    net1 = build_ssd('test', 300, num_classes_waterline)            # initialize SSD
    net1.load_state_dict(torch.load(args.trained_model_waterline))
    net1.eval()
    print('Finished loading model!')
    # load data
    dataset1 = customDetection(args.custom_root, [('gauge', set_type)], None, customAnnotationTransform(class_to_ind=dict(zip(CUSTOM_CLASSES_GAUGE, range(len(CUSTOM_CLASSES_GAUGE))))))
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(args.save_folder, net, args.cuda, dataset1,
             BaseTransform(net.size, (104, 117, 123)),
             thresh=args.visual_threshold, labelmap=labelmap_gauge)

    rootPath = 'F:/ssd/data/video/gauge'
    rootPath_temp = 'F:/ssd/data/video/test'
    imgList_gauge = {}

    with open(os.path.join(args.save_folder, 'result_gauge.txt'), 'r') as f:
        text_lines = f.readlines()
        for line in text_lines:
            info = line.split(" ")
            name, score, x1, y1, x2, y2 = info
            if name in imgList_gauge:
                if float(score) > imgList_gauge[name]['score']:
                    imgList_gauge[name] = {
                        'score': float(score),
                        'x1': float(x1),
                        'y1': float(y1),
                        'x2': float(x2),
                        'y2': float(y2)
                    }
            else:
                imgList_gauge[name] = {
                    'score': float(score),
                    'x1': float(x1),
                    'y1': float(y1),
                    'x2': float(x2),
                    'y2': float(y2)
                }

    img_path = os.path.join(rootPath, 'JPEGImages', '%s.jpg')
    devkit_imgpath = os.path.join(get_output_dir(devkit_imgpath), '%s.jpg')
    devkit_imgsetpath = os.path.join(get_output_dir(devkit_imgsetpath), '%s.txt')
    devkit_annopath = os.path.join(get_output_dir(devkit_annopath), '%s.xml')
    with open(devkit_imgsetpath % ('test'), 'w') as f:
        for obj in imgList_gauge.items():
            name, img = obj
            image = cv2.imread(img_path % name)
            (h, w, c) = image.shape
            x1 = max(math.floor(img['x1']), 0)
            y1 = max(math.floor(img['y1']), 0)
            x2 = min(math.floor(img['x2']), w)
            y2 = min(math.floor(img['y2']), h)
            if DEBUG:
                cv2.rectangle(image, (x1, y1), (x2, y2), (255,0,0), 5)
                image = cv2.resize(image, (512, 512))
                cv2.imshow('w1', image)
                cv2.waitKey()
            else:
                image = image[y1:y2, x1:x2]
                # cv2.imshow('w1', image)
                cv2.imwrite(devkit_imgpath % name, image, [100])
                f.write(name + '\n')
                # cv2.waitKey()
            with open(devkit_annopath % (name), 'w') as f_a:
                f_a.write(xmlData(name, x2 - x1, y2 - y1, 'waterline'))
    
    dataset2 = customDetection(args.custom_root, [('test', set_type)], None, customAnnotationTransform(class_to_ind=dict(zip(CUSTOM_CLASSES_WATERLINE, range(len(CUSTOM_CLASSES_WATERLINE))))))

    if args.cuda:
        net1 = net1.cuda()
        cudnn.benchmark = True
    
    # evaluation
    test_net(args.save_folder, net1, args.cuda, dataset2,
             BaseTransform(net.size, (104, 117, 123)),
             thresh=args.visual_threshold, labelmap=labelmap_waterline)

    imgList_waterline = {}
    with open(os.path.join(args.save_folder, 'result_waterline.txt'), 'r') as f:
        text_lines = f.readlines()
        for line in text_lines:
            info = line.split(" ")
            name, score, x1, y1, x2, y2 = info
            if name in imgList_waterline:
                if float(score) > imgList_waterline[name]['score']:
                    imgList_waterline[name] = {
                        'score': float(score),
                        'x1': float(x1),
                        'y1': float(y1),
                        'x2': float(x2),
                        'y2': float(y2)
                    }
            else:
                imgList_waterline[name] = {
                    'score': float(score),
                    'x1': float(x1),
                    'y1': float(y1),
                    'x2': float(x2),
                    'y2': float(y2)
                }
    
    imgList_mark = {}
    with open(os.path.join(args.save_folder, 'result_mark.txt'), 'r') as f:
        text_lines = f.readlines()
        for line in text_lines:
            info = line.split(" ")
            name, score, x1, y1, x2, y2 = info
            if name in imgList_mark:
                if float(score) > imgList_mark[name]['score']:
                    imgList_mark[name] = {
                        'score': float(score),
                        'x1': float(x1),
                        'y1': float(y1),
                        'x2': float(x2),
                        'y2': float(y2)
                    }
            else:
                imgList_mark[name] = {
                    'score': float(score),
                    'x1': float(x1),
                    'y1': float(y1),
                    'x2': float(x2),
                    'y2': float(y2)
                }
    
    cv2.namedWindow('w2',1)
    use_origin = True

    if not use_origin:
        img_path = os.path.join(rootPath_temp, 'JPEGImages', '%s.jpg')
    count = 0
    for name in imgList_gauge:
        img_gauge = imgList_gauge[name]
        img_waterline = imgList_waterline[name]
        img_mark = imgList_mark[name]
        
        if not use_origin:
            image = cv2.imread(img_path % name)
            (h, w, c) = image.shape

            x1_w = max(math.floor(img_waterline['x1']), 0)
            y1_w = max(math.floor(img_waterline['y1']), 0)
            x2_w = min(math.floor(img_waterline['x2']), w)
            y2_w = min(math.floor(img_waterline['y2']), h)

            x1_m = max(math.floor(img_mark['x1']), 0)
            y1_m = max(math.floor(img_mark['y1']), 0)
            x2_m = min(math.floor(img_mark['x2']), w)
            y2_m = min(math.floor(img_mark['y2']), h)
            
            cv2.rectangle(image, (x1_w, y1_w), (x2_w, y2_w), (255,0,0), 5)
            cv2.rectangle(image, (x1_m, y1_m), (x2_m, y2_m), (0,255,0), 5)
            image = cv2.resize(image, (512, 512))
            cv2.imshow('w2', image)
            cv2.waitKey()
        else:
            image = cv2.imread(img_path % name)
            (h, w, c) = image.shape

            x1_g = math.floor(img_gauge['x1'])
            y1_g = math.floor(img_gauge['y1'])
            x2_g = math.floor(img_gauge['x2'])
            y2_g = math.floor(img_gauge['y2'])

            x1_w = max(math.floor(img_waterline['x1']), 0)
            y1_w = max(math.floor(img_waterline['y1']), 0)
            x2_w = min(math.floor(img_waterline['x2']), w)
            y2_w = min(math.floor(img_waterline['y2']), h)

            x1_m = max(math.floor(img_mark['x1']), 0)
            y1_m = max(math.floor(img_mark['y1']), 0)
            x2_m = min(math.floor(img_mark['x2']), w)
            y2_m = min(math.floor(img_mark['y2']), h)
            
            if (y1_w + y2_w) > (y1_m + y2_m):
                count += 1

            cv2.rectangle(image, (x1_g, y1_g), (x2_g, y2_g), (255,0,0), 5)
            cv2.rectangle(image, (x1_g + x1_w, y1_g + y1_w), (x1_g + x2_w, y1_g + y2_w), (0,255,0), 5)
            cv2.rectangle(image, (x1_g + x1_m, y1_g + y1_m), (x1_g + x2_m, y1_g + y2_m), (0,0,255), 5)
            
            image = cv2.resize(image, (512, 512))
            cv2.putText(image, 'gauge: %.2f' % img_gauge['score'], (10, 40), cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 0, 0), 2)
            cv2.putText(image, 'waterline: %.2f' % img_waterline['score'], (10, 80), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 255, 0), 2)
            cv2.putText(image, 'mark: %.2f' % img_mark['score'], (10, 120), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 255), 2)
            cv2.imshow('w2', image)
            cv2.waitKey()
    print('correct count:', count)

if __name__ == '__main__':
    test_custom()
