# -*- coding: utf-8 -*-
"""
Created on Wed Mar 09 20:46:27 2016

@author: tanfan.zjh
"""
import glob

class Node:
    def __init__(self,word):
        self.word = word
        self.index = -1 #start from 1
        self.label = 0 # event type
        self.parent_index = -1
        self.childs_index = []
        self.childs_word = []
        self.parent_word = None
        self.dependency = None
        self.start = 0
        self.end = 0
    def is_leaf(self):
        return self.childs_index == None or len(self.childs_index) == 0
    def is_root(self):
        return self.parent_index == 0     
    def __repr__(self):
        str_childs_index = [str(index) for index in self.childs_index]
        return '['+str(self.index)+',('+str(self.parent_index)+'),('+\
                ','.join(str_childs_index)+')]'
    
input_dir = 'MLEE_DATA/train-gdep-trigger-candidate'

##2	neovascularization	neovascularization	I-NP	NN	O	0	ROOT	3	neovascularization
tree_container = []
for f in glob.glob(input_dir+'/*.ann'):
    input = open(f)
    node_container = []
    for line in input.readlines():
        if len(line)==0 or len(line.split('\t'))<5:
            for node in node_container:
                if node.label>0:
                    tree_container.append(node_container)
                    break
            node_container = []
            continue
        toks = line.split('\t')
        node = Node(toks[1])
        node.index = int(toks[0])
        node.parent_index = int(toks[6])
        node.label = int(toks[8])
        node.dependency = toks[7]
        node_container.append(node)
    input.close()

for tree in tree_container:
    for node1 in tree:
        for node2 in tree:
            if node2.parent_index == node1.index:
                node1.childs_index.append(node2.index)

#1338 sentences contain triggers
#print len(tree_container)

#1758 sentences
#print len(tree_container)

###############################################################################
import copy
def build_sub_tree(tree,node_index):
    sub_tree = []
    node_copy = None
    for node in tree:
        if node.index == node_index:
            node_copy = copy.deepcopy(node)
            sub_tree.append(node_copy)
    if node_copy.is_leaf():
        return sub_tree
    else:
        for index in node_copy.childs_index:
            sub_tree.extend(build_sub_tree(tree,index))
        return sub_tree

#print tree_container[0]
#print build_sub_tree(tree_container[0],4)

## [advmod(develop-4, How-1), aux(develop-4, did-2),
def tree_to_string(tree):
    tree_str = []
    tree_str.append('[')
    def _parent_word_by_index(index):
        if index == 0:
            return 'ROOT'
        for node in tree:
            if node.index == index:
                return node.word
        return 'ROOT'
    for node in tree:
        tree_str.append(node.dependency)
        tree_str.append('(')
        tree_str.append(node.word)
        tree_str.append('-')
        tree_str.append(str(node.index))
        tree_str.append(', ')
        tree_str.append(_parent_word_by_index(node.parent_index))
        tree_str.append('-')
        tree_str.append(str(node.parent_index))
        tree_str.append(')')
        tree_str.append(', ')
    tree_str = tree_str[:-1]
    tree_str.append(']')
    return ''.join(tree_str)
#print tree_to_string(tree_container[0])

labeld_tree = {}
label_file = open('MLEE_DATA/mlee-train-labels.txt','w')
example_file = open('MLEE_DATA/mlee-train-data.txt','w')
for tree in tree_container:
    for node in tree:
        if node.label < 0:
            continue
        if node.is_root():# root is trigger
            label_file.write(str(node.label)+'\n')
            example_file.write(tree_to_string(tree)+'\n') # tree:tree_container
        elif node.is_leaf():
            label_file.write(str(node.label)+'\n')
            example_file.write(tree_to_string(build_sub_tree(tree,node.parent_index))+'\n')
        else:
            label_file.write(str(node.label)+'\n')
            example_file.write(tree_to_string(build_sub_tree(tree,node.index))+'\n')
            
label_file.close()
example_file.close()

###############################################################################

if __name__=='__main__':
    for f in glob.glob(input_dir+'/*.ann'):
        pass

'''
input_file = open('test_ann')
node_container = []
for line in input_file.readlines():
    toks = line.split('\t')
    node = Node(toks[1])
    node.index = int(toks[0])
    node.parent_index = int(toks[6])
    node.label = int(toks[8])
    node.depdendency = toks[7]
    node_container.append(node)
input_file.close()

for node1 in node_container:
    for node2 in node_container:
        if node2.parent_index == node1.index:
            node1.childs_index.append(node2.index)
'''