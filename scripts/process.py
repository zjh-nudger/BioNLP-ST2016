#coding:utf-8
"""
Created on Mon Mar 07 14:24:32 2016

@author: tanfan.zjh
"""

import path,glob

for fn in glob.glob(path.SOURCE_DATA_BINARY_TRAIN+'/*.txt'):
    print fn
    for f in open(fn):
        print f

    break