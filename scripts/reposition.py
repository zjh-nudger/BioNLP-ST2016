# -*- coding: utf-8 -*-
"""
Created on Tue Mar 08 14:21:53 2016

sentence split以及tokenization之后标注重定位

@author: tanfan.zjh
"""

import codecs

d = 'C:/Users/tanfan.zjh/Desktop'

ss_file = open(d+'/test1_ss.txt')
src_file = codecs.open(d+'/test1_src.txt','r','utf-8')
a1_file = open(d+'/test1.a1')


txt = ''.join([line for line in src_file])
ss_txt = ''.join([line for line in ss_file])

span_dic = {}
for line in a1_file:
    if ';' in line:
        continue
    toks = line.strip().split('\t')
    toks1 = toks[1].split()
    start = int(toks1[1])
    end = int(toks1[2])
    span_dic[(start,end)] = toks[0]


