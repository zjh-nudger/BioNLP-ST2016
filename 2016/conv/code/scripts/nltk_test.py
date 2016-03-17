# -*- coding: utf-8 -*-
"""
Created on Sun May 24 13:50:02 2015

@author: zjh
"""

import sys,nltk,os
#s='I am a students.'

#r=nltk.word_tokenize(s)

root=sys.argv[1]
output=open(sys.argv[2],'w')
for parent,dirnames,filenames in os.walk(root):
    for fn in filenames:
        f=open(parent+'/'+fn)
        for line in f:
            s=nltk.word_tokenize(line)
            new_s=' '.join(s)
            output.write(new_s)
            output.write('\n')
        f.close()
output.close()
'''
output=open(sys.argv[1],'w')
for i,w in enumerate(sys.stdin):
    ww=w.split('\n')
    for x in ww:
        s=nltk.word_tokenize(x)
        new_s=' '.join(s)
        output.write(new_s)
        output.write('\n')

output.close()
'''