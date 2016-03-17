# -*- coding: utf-8 -*-
"""
Created on Wed Feb 03 07:58:20 2016

@author: tanfan.zjh
"""

import logging,os,re,time
from gensim.models import Word2Vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

class SentenceIterator(object):
    def __init__(self,train_data_dir):
        self.train_data_dir = train_data_dir
    def __iter__(self):
        for fn in os.listdir(self.train_data_dir):
            f = open(os.path.join(self.train_data_dir,fn))
            for line in f:
                yield line.split()
'''
class gensim.models.word2vec.Word2Vec(sentences=None, size=100, alpha=0.025,
 window=5, min_count=5, max_vocab_size=None, sample=0.001, seed=1, workers=3, 
 min_alpha=0.0001, sg=0, hs=0, negative=5, cbow_mean=1, hashfxn=<built-in function hash>, 
 iter=5, null_word=0, trim_rule=None, sorted_vocab=1, batch_words=10000)
'''
if __name__=='__main__':
    sentences = SentenceIterator('train_data')
    model = Word2Vec(sentences,min_count=5,workers=4,size=200,
                     sample=1e-4,window=5,sg=0)
    model.save_word2vec_format('wordvec_200.txt',binary=False)
    