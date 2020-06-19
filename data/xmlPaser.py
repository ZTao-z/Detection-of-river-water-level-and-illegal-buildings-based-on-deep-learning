#!/usr/bin/python
# -*- coding: UTF-8 -*-
 
import xml.sax
from pathlib import Path, PurePath

total = {}
 
class MovieHandler( xml.sax.ContentHandler ):
    def __init__(self):
        self.CurrentData = ""
        self.name = ''
 
    # 元素开始事件处理
    def startElement(self, tag, attributes):
        self.CurrentData = tag
 
    # 元素结束事件处理
    def endElement(self, tag):
        if self.CurrentData == "name":
            if self.name in total:
                total[self.name] += 1
            else:
                total[self.name] = 1
        self.CurrentData = ""
 
    # 内容事件处理
    def characters(self, content):
        if self.CurrentData == "name":
            self.name = content

if ( __name__ == "__main__"):
   
    # 创建一个 XMLReader
    parser = xml.sax.make_parser()
    # turn off namepsaces
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)

    # 重写 ContextHandler
    Handler = MovieHandler()
    parser.setContentHandler( Handler )

    path = '.\\piaofu\\piao\\shenhe\\Annotations'
    p = Path(path)
    files = [x for x in p.iterdir() if x.is_file()]
    for f in files:
        parser.parse(path+'\\'+f.name)
    print(total)