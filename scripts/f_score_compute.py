# -*- coding: utf-8 -*-
"""
Created on Wed Mar 09 19:05:42 2016

@author: tanfan.zjh
"""

predict = [int(line.strip()) for line in open('tmp_file/pred.txt')]
real = [int(line.strip()) for line in open('tmp_file/real.txt')]

true_pos = 0
pos = 0
real_pos = 0
typex = 185
for p,r in zip(predict,real):
    if p == r and p == typex:
        true_pos += 1
    if p == typex:
        pos += 1
    if r == typex:
        real_pos += 1
precision = float(true_pos) / (float(pos)+0.0000000001)
recall = float(true_pos) / float(real_pos)
print recall

#print 2*(precision*recall) / (precision+recall)