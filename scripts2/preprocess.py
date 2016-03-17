# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 15:49:09 2016

@author: tanfan.zjh
"""
import re,codecs
'''
# CGNCCA	CGNCCA	NN	B-NP	O
data = codecs.open('unlabeled_data/ab.pos.gt',encoding='utf-8')
train_data = codecs.open('train_data/ab.txt','w',encoding='utf-8')
new_line = []
print 'tt'

for line in data:
    toks = line.split('\t')
    if len(toks) == 1:
        train_data.write(' '.join(new_line)+'\n')
        new_line = []
        continue
    else:
        new_toks = re.split('/|-',toks[0].lower())
        for t in new_toks:
            if t == '':
                continue
            try:
                new_t = codecs.decode(t,'utf-8').encode('utf-8')
                new_line.append(new_t)
            except:
                #print t
                continue
            

train_data.close()
data.close()
'''

import glob,codecs
train_data = codecs.open('train_data/ab-devel.txt','w',encoding='utf-8')
for fn in glob.glob('C:/Users/tanfan.zjh/OneDrive/spyder workspace/BioNLP-ST2016/supporting resources/genia-tagger/dev/*.gt'):
    f = codecs.open(fn,encoding='utf-8')
    new_line = []
    for line in f:
        toks = line.split('\t')
        if len(toks) == 1:
            train_data.write(' '.join(new_line)+'\n')
            new_line = []
            continue
        else:
            try:
                tt = codecs.decode(toks[0],'utf-8').encode('utf-8')
                new_line.append(tt)
            except:
                continue
            
    f.close()
train_data.close()
