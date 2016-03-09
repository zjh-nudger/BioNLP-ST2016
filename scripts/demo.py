# -*- coding: utf-8 -*-
"""
Created on Tue Mar 08 21:18:32 2016

@author: tanfan.zjh
"""
c = 0
for line in open('dev'):
    toks = line.split()
    typex = int(toks[0])
    if typex>0:
        c+=1
print c
print 1628-1541
