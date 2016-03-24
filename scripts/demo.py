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
'''
import numpy
a = [[1,2],[3,4],[5,6]]
b = [[.1,.2],[.3,.4],[.5,.6]]
c = numpy.random.permutation(numpy.append(a,b,axis=1))
print c[:,numpy.shape(a)[1]:]



try:
    import matplotlib.pyplot as plt
except:
    raise

import networkx as nx

G=nx.house_graph()
# explicitly set positions
pos={0:(0,0),
     1:(1,0),
     2:(0,1),
     3:(1,1),
     4:(0.5,2.0)}

nx.draw_networkx_nodes(G,pos,node_size=2000,nodelist=[4])
nx.draw_networkx_nodes(G,pos,node_size=3000,nodelist=[0,1,2,3],node_color='b')
nx.draw_networkx_edges(G,pos,alpha=0.5,width=6)
plt.axis('on')
plt.savefig("house_with_colors.png") # save as png
plt.show() # display
'''
'''
f = open('data/train')
c=0
for line in f:
    toks = line.strip().split()
    if int(toks[0])>0:
        c += 1
f.close()
print c
for i in xrange(2,5):
    print i
'''
f = open('data/train_data/train')
max_l = 0
for line in f:
    toks = line.strip().split()
    typex = int(toks[0])
    if typex == 19:
        if len(toks[1:]) > max_l:
            max_l = len(toks[1:])
print max_l
f.close()