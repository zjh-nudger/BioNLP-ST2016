# -*- coding: utf-8 -*-
"""
Created on Tue Mar 08 21:18:32 2016

@author: tanfan.zjh
"""
'''
import theano.tensor as T
import numpy as np

x_s = [[1,2],
       [3,4]]
x = np.asarray(x_s)
w = [[0.1,0.1,0.1],
     [0.2,0.1,0.1],
     [0.3,0.1,0.1],
     [0.4,0.1,0.1],
     [0.5,0.1,0.1]]
pos = [[0.01,0.02],
       [0.02,0.03]]
Words = np.asarray(w)
pos_emb = np.asarray(pos)

pos_x_1 = [[0,0],
         [1,1]]
pos_x = np.asarray(pos_x_1)
#layer0_input = Words[T.cast(x.flatten(),dtype="int32")].\
#                    reshape((x.shape[0],1,x.shape[1],Words.shape[1]))
layer0_input = Words[x.flatten()].\
                    reshape((x.shape[0],1,x.shape[1],Words.shape[1]))
pos = pos_emb[pos_x.flatten()].\
             reshape((pos_x.shape[0],1,pos_x.shape[1],pos_emb.shape[1]))

f = T.concatenate([layer0_input,pos],axis=3)
print f.eval()
'''
import numpy
a = [[1,2],[3,4],[5,6]]
b = [[.1,.2],[.3,.4],[.5,.6]]
c = numpy.random.permutation(numpy.append(a,b,axis=1))
print c[:,numpy.shape(a)[1]:]
