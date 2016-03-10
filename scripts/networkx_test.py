# -*- coding: utf-8 -*-
"""
Created on Wed Mar 09 21:35:11 2016

@author: tanfan.zjh
"""
class Word:
    def __init__(self,word,lemma,pos,start,end):
        self.word = word
        self.lemma = lemma
        self.pos = pos
        self.start = start
        self.end = end
        self.parent = None
    def __repr__(self):
#        return self.word+'\t'+self.lemma+'\t'+self.pos+'\t'+\
#                str(self.start)+'\t'+str(self.end)
        return self.word
    def _str__(self):
        return self.word
    def __eq__(self,other):
        return self.start == other.start and self.end == other.end and self.word == other.word
    def __hash__(self):
        return hash(self.word)+hash(self.start)+hash(self.end)

import networkx as nx



w1 = Word('1','','',1,2)
w2 = Word('2','','',3,6)
w3 = Word('3','','',8,9)

graph = nx.Graph([(w1,w2),(w2,w3)])

print graph.nodes()


#edges = [(10,8),(10,23),(10,9),(10,13),(8,1),(8,4),
#         (4,2),(4,3),(4,6),(6,5),(6,7),(13,11),(13,12),
#         (13,16),(13,17),(13,19),(16,14),(16,15),
#         (19,18),(19,22),(22,20),(22,21)]
#print len(edges)

#graph.add_edges_from(edges)

#print nx.shortest_path(graph,7,14)
