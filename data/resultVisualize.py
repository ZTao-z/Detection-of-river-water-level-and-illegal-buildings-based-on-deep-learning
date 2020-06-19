import cv2
import numpy as np
import os.path as osp
import math

rootPath = 'F:/ssd/data/video/waterline'

imgList = {}

if __name__ == "__main__":
    with open('./det_test_waterline_99.txt', 'r') as f:
        text_lines = f.readlines()
        for line in text_lines:
            info = line.split(" ")
            name, score, x1, y1, x2, y2 = info
            if name in imgList:
                if float(score) > imgList[name]['score']:
                    imgList[name] = {
                        'score': float(score),
                        'x1': float(x1),
                        'y1': float(y1),
                        'x2': float(x2),
                        'y2': float(y2)
                    }
            else:
                imgList[name] = {
                    'score': float(score),
                    'x1': float(x1),
                    'y1': float(y1),
                    'x2': float(x2),
                    'y2': float(y2)
                }

    cv2.namedWindow('w1',1)
    img_path = osp.join(rootPath, 'JPEGImages', '%s.jpg')
    for obj in imgList.items():
        name, img = obj
        image = cv2.imread(img_path % name)
        (h, w, c) = image.shape
        cv2.rectangle(image, (math.floor(img['x1']), math.floor(img['y1'])), (math.floor(img['x2']), math.floor(img['y2'])), (255,0,0), 5)
        # cv2.putText(image, img['score'], (math.floor(img['x1']), math.floor(img['y1'])), cv2.FONT_HERSHEY_COMPLEX, 5, (0, 255, 0), 12)
        # sc = min(512, h) / h
        # image = cv2.resize(image, (math.floor(w * sc), math.floor(h * sc)))
        image = cv2.resize(image, (512, 512))
        cv2.imshow('w1', image)
        cv2.waitKey()
