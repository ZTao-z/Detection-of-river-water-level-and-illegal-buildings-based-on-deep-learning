#!/usr/bin/python
# -*- coding: UTF-8 -*-
 
import xml.sax
from pathlib import Path, PurePath

r = {}

result = []

# label = {
#     'garbage': 0,
#     'garbagew': 1,
#     'www': 2,
#     'w': 3
# }

label = {
    'waterline': 0,
}
 
class MovieHandler( xml.sax.ContentHandler ):
    def __init__(self):
        self.tag = ""
        self.boxes = []
        self.box = {
            'name': '',
            'xmin': 0,
            'xmax': 0,
            'ymin': 0,
            'ymax': 0
        }
        self.size = {
            'width': 0,
            'height': 0,
            'depth': 0
        }
 
    # 元素开始事件处理
    def startElement(self, tag, attributes):
        self.tag = tag
 
    # 元素结束事件处理
    def endElement(self, tag):
        if self.tag == 'depth':
            r['data']['size'] = self.size
        if self.tag == 'ymax':
            r['data']['boxes'].append(self.box)
        self.tag = ""
 
    # 内容事件处理
    def characters(self, content):
        if self.tag == 'size':
            self.size = {
                'width': 0,
                'height': 0,
                'depth': 0
            }
        elif self.tag == 'object':
            self.box = {
                'name': '',
                'xmin': 0,
                'xmax': 0,
                'ymin': 0,
                'ymax': 0
            }
        elif self.tag == 'width':
            self.size['width'] = int(content)
        elif self.tag == 'height':
            self.size['height'] = int(content)
        elif self.tag == 'depth':
            self.size['depth'] = int(content)
        elif self.tag == 'name':
            self.box['name'] = content
        elif self.tag == 'xmin':
            self.box['xmin'] = int(content)
        elif self.tag == 'xmax':
            self.box['xmax'] = int(content)
        elif self.tag == 'ymin':
            self.box['ymin'] = int(content)
        elif self.tag == 'ymax':
            self.box['ymax'] = int(content)        

if ( __name__ == "__main__"):
   
    # 创建一个 XMLReader
    parser = xml.sax.make_parser()
    # turn off namepsaces
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)

    # 重写 ContextHandler
    Handler = MovieHandler()
    parser.setContentHandler( Handler )

    path = './video/waterline/Annotations'
    p = Path(path)
    files = [x for x in p.iterdir() if x.is_file()]
    for f in files:
        r = {
            'file': f.name[0: -4],
            'data': {
                'size': {},
                'boxes': []
            }
        }
        parser.parse(path+'/'+f.name)
        result.append(r)
    for r in result:
        # with open(".\\labels\\" + r['file'] + ".txt", "w") as f:
        width = r['data']['size']['width']
        height = r['data']['size']['height']
        for b in r['data']['boxes']:
            center_x = (b['xmax'] + b['xmin']) / 2 / width
            center_y = (b['ymax'] + b['ymax']) / 2 / height
            width_x = (b['xmax'] - b['xmin']) / width
            height_y = (b['ymax'] - b['ymin']) / height
            label_idx = label[b['name']]
            if width_x == 0 or height_y == 0 or (b['name'] != 'waterline'):
                print(r['file'])
                break
            # f.write(str(label_idx) + ' ' + str(center_x) + ' ' + str(center_y) + ' ' + str(width_x) + ' ' + str(height_y) + "\n")