# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 18:20:24 2016

@author: tanfan.zjh
"""

import gensim

'''
(corpus=None, num_topics=100, id2word=None, distributed=False, 
chunksize=2000, passes=1, update_every=1, alpha='symmetric', eta=None,
 decay=0.5, offset=1.0, eval_every=10, iterations=50, gamma_threshold=0.001,
 minimum_probability=0.01)
'''

train = open('ab-devel.txt')
documents = []
for line in train:
    documents.append([word.lower() for word in line.split()])
train.close()

dictionary = gensim.corpora.Dictionary(documents)
corpus = [dictionary.doc2bow(text) for text in documents]

lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, 
                                      num_topics=5, update_every=1, 
                                      chunksize=10000, passes=1)
lda.save('lda.model')
