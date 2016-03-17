# -*- coding: utf-8 -*-
"""
Created on Tue Mar 08 19:47:22 2016

提取实体上下文信息

@author: tanfan.zjh
"""
import sys
sys.path.append('..')

import glob,path,re,project,itertools
from collections import OrderedDict
import networkx as nx

class Word(object):
    def __init__(self,word,lemma,pos,start,end):
        self.word = word
        self.lemma = lemma
        self.pos = pos
        self.start = start
        self.end = end
    def __repr__(self):
#        return self.word+'\t'+self.lemma+'\t'+self.pos+'\t'+\
#                str(self.start)+'\t'+str(self.end)
        return self.word
    def __eq__(self,other):
        return self.start == other.start and self.end == other.end and self.word == other.word
    def __hash__(self):
        return hash(self.word)+hash(self.start)+hash(self.end)

class TreeNode(Word):
    def __init__(self,index,word,lemma,pos,parent,dependency,start,end):
        super(TreeNode,self).__init__(word,lemma,pos,start,end)
        self.index = index
        self.parent = parent
        self.dependency = dependency
    def __repr__(self):
        return str(self.index)+' '+self.word+' '+str(self.start)+' '+str(self.end)+'\n'
    def __eq__(self,other):
        return self.index == other.index and self.start == other.start and \
                self.end == other.end and self.word == other.word
    def __hash__(self):
        return hash(self.index)+hash(self.parent)+hash(self.word)+\
               hash(self.start)+hash(self.end)
        
class Span:
    def __init__(self,start,end):
        self.start = start
        self.end = end
    def __repr__(self):
        return str(self.start)+' '+str(self.end)

class EntityAnnotation:
    def __init__(self,idx,typex,spans,name):
        self.idx = idx
        self.typex = typex
        self.spans = spans # list of span
        self.name = name
    def __repr__(self):
        spans = [str(span) for span in self.spans]
        return self.idx+'\t'+self.typex+' '+';'.join(spans)+'\t'+self.name
    def start(self):
        starts = []
        for span in self.spans:
            starts.append(span.start)
        return min(starts)
    def end(self):
        ends = []
        for span in self.spans:
            ends.append(span.end)
        return max(ends)
        
class RelationAnnotation:
    def __init__(self,idx,typex,element1_name,element1_idx,element2_name,element2_idx):
        self.idx = idx
        self.typex = typex
        self.element1_name = element1_name
        self.element1_idx = element1_idx
        self.element2_name = element2_name
        self.element2_idx = element2_idx
    def __repr__(self):
        return self.idx + '\t'+self.typex+'\t'+self.element1_name+':'+self.element1_idx+'\t'+\
               self.element2_name+':'+self.element2_idx

def get_a1_annotations(a1_file):
    annotations = OrderedDict()
    for line in a1_file:
        toks = line.strip().split('\t')
        toks1 = re.split(' |;',toks[1])
        span_list = []
        for start,end in zip(toks1[1::2],toks1[2::2]):
            span_list.append(Span(int(start),int(end)))
        annotation = EntityAnnotation(toks[0],toks1[0],span_list,toks[2])
        annotations[toks[0]] = annotation
    return annotations

def get_sentences_from_gt(gt_file):
    document = []
    sentence = []   
    for line in gt_file:
        toks = re.split('\t',line.strip())
        toks = [tok for tok in toks if tok!='']
        if not len(line) > 2:
            document.append(sentence)
            sentence = []
        else:
            word = Word(toks[0],toks[1],toks[2],int(toks[5]),int(toks[6]))
            sentence.append(word)
    return document
    
def get_a2_annotations(a2_file):
    annotations = {}
    for line in a2_file:
        toks = re.split('\t|:| ',line.strip())
        annotation = RelationAnnotation(toks[0],toks[1],toks[2],toks[3],toks[4],toks[5])
        annotations[(toks[3],toks[5])] = annotation
    return annotations

def get_context(e1_start_index,e1_end_index,e2_start_index,e2_end_index,sentence,window):
    selected_index = set()
    '''
    for i in xrange(e1_start_index,e1_end_index+1):
        selected_index.add(i)
    for i in xrange(e2_start_index,e2_end_index+1):
        selected_index.add(i)
    for i in xrange(e1_end_index,e2_start_index+1):
        selected_index.add(i)
    '''
    '''
    for i in xrange(e1_start_index,e2_end_index+1):
        selected_index.add(i)
    for i in xrange(e2_start_index,e1_end_index+1):
        selected_index.add(i)
    '''
    '''
    for i in xrange(window):
        index_before_e1 = e1_start_index - i - 1
        index_after_e1 = e1_end_index + i +1
        index_before_e2 = e2_start_index - i - 1
        index_after_e2 = e2_end_index + i + 1
        selected_index.add(index_after_e1)
        selected_index.add(index_after_e2)
        selected_index.add(index_before_e1)
        selected_index.add(index_before_e2)
    '''
    for i in xrange(e1_end_index+1,e2_start_index):
        selected_index.add(i)
    for i in xrange(e2_end_index+1,e1_start_index):
        selected_index.add(i)
        
    selected_index_edge = []
    for index in selected_index:
        if index < 0 or index >= len(sentence):
            print 'error'
            continue
        selected_index_edge.append(index)
    selected_index = list(selected_index_edge)
    list.sort(selected_index)
    contexts = []
    for index in selected_index:
        contexts.append(sentence[index])
    return contexts    

# return the index of e1\e2 in the sentence  
def get_index(element1,element2,sentence):
    e1_start_index = 0
    e1_end_index = 0
    e2_start_index = 0
    e2_end_index = 0
    for word in sentence:
        if word.start <= element1.start() and word.end > element1.start():
            e1_start_index = sentence.index(word)
        if word.end >= element1.end() and word.start < element1.end():
            e1_end_index = sentence.index(word)
        if word.start <= element2.start() and word.end > element2.start():
            e2_start_index = sentence.index(word)
        if word.end >= element2.end() and word.start < element2.end():
            e2_end_index = sentence.index(word)
    #assert len(sentencex) != 0
    return e1_start_index,e1_end_index,e2_start_index,e2_end_index

def get_dependency_graph(gdep_file):
    graphs = []
    tree_node_list = []
    for line in gdep_file:
        toks = line.strip().split('\t')
        #print word_list
        if len(toks) < 3:
            #print word_list
            graph = nx.Graph()
            graph.add_nodes_from(tree_node_list)
            edge_list = []
            for word in tree_node_list:
                if word.parent - 1 == -1:
                    continue
                edge_list.append((tree_node_list[word.parent-1],word))
            graph.add_edges_from(edge_list)
            #print str(len(tree_node_list))+' '+str(len(graph.edges()))
            graphs.append(graph)
            tree_node_list = []
        else:
            assert len(toks) == 10
            ## index,word,lemma,pos,parent,dependency,start,end
            ## 1	The	The	B-NP	DT	O	8	NMOD
            node = TreeNode(int(toks[0]),toks[1],toks[2],toks[3],
                            int(toks[6]),toks[7],int(toks[8]),int(toks[9]))
            tree_node_list.append(node)
    return graphs

def get_shortest_path(entity_node1,entity_node2,graph):
    nodes = itertools.product(entity_node1,entity_node2)
    shortest_path = []
    for source,target in nodes:
        if source.index < target.index:
            shortest_path.append(nx.shortest_path(graph,source,target)[1:-1])
        else:
            shortest_path.append(nx.shortest_path(graph,target,source)[1:-1])
#    if len(shortest_path) == 0:
#        print entity_node1
#        print entity_node2
    sp = shortest_path[0]
    for _sp in shortest_path:
        if len(_sp) < len(sp):
            sp = _sp
    sp_word = []
    for node in entity_node1:
        sp_word.append(node.word)
    for node in sp:
        sp_word.append(node.word)
    for node in entity_node2:
        sp_word.append(node.word)
    return sp_word

def create_shortest_path_examples(gt_file,a1_file,a2_file,gdep_file,output_file):
    document = get_sentences_from_gt(gt_file)
    entities = get_a1_annotations(a1_file)
    relations = get_a2_annotations(a2_file)
    dependency_graphs = get_dependency_graph(gdep_file)
    index = 0    
    for sentence in document:
        sentence_entity_set = set()
        graph = dependency_graphs[index]
        index += 1
        for word in sentence:
            for idx,entity in entities.iteritems():
                if word.start <= entity.start() and word.end > entity.start():
                    sentence_entity_set.add(entity.idx)
        for combination in itertools.product(sentence_entity_set,sentence_entity_set):
            if relations.has_key(combination):           
                a2_annotation = relations[combination]
                if not a2_annotation.typex == 'Composes_Primary_Structure':
                    continue
                relations.pop(combination)
                typex_id = 1 #project.RELATIONS.index(a2_annotation.typex) + 1
                element1 = entities[combination[0]]
                element2 = entities[combination[1]]
            else: # negative examples
                typex_id = 0
                element1 = entities[combination[0]]
                element2 = entities[combination[1]]
                if not (element1.typex in project.Composes_Primary_Structure['DNA_Part'] and \
                   element2.typex in project.Composes_Primary_Structure['DNA']):
                    continue
            #entity_type_1 = element1.typex
            #entity_type_2 = element2.typex
            entity_node1 = []
            entity_node2 = []
            for node in graph.nodes():
                if (node.start >= element1.start() and node.end <= element1.end()) or\
                   (node.start <= element1.start() and node.end >= element1.end()):
                    entity_node1.append(node)
                if (node.start >= element2.start() and node.end <= element2.end()) or\
                   (node.start <= element2.start() and node.end >= element2.end()):
                    entity_node2.append(node)
            shortest_path = get_shortest_path(entity_node1,entity_node2,graph)
            output_file.write(str(typex_id)+' ' + ' '.join(shortest_path)+'\n')    

a_i = 0
def create_examples(gt_file,a1_file,a2_file,output_file,window):
    global a_i
    document = get_sentences_from_gt(gt_file)
    entities = get_a1_annotations(a1_file)
    relations = get_a2_annotations(a2_file)
    o = open('Composes_Protein_Complex.txt','a')
    ind = 0
    for sentence in document:
        s = {}
        sentence_entity_set = set()
        for word in sentence:
            for idx,entity in entities.iteritems():
                if word.start <= entity.start() and word.end > entity.start():
                    sentence_entity_set.add(entity.idx)
        for combination in itertools.product(sentence_entity_set,sentence_entity_set):
            if combination[0] == combination[1]:
                continue
            if relations.has_key(combination):           
                a2_annotation = relations[combination]
                if (not a2_annotation.typex == 'Composes_Protein_Complex') and\
                   (not a2_annotation.typex == 'Composes_Primary_Structure') and\
                   (not a2_annotation.typex == 'Is_Member_Of_Family') and \
                   (not a2_annotation.typex == 'Is_Protein_Domain_Of'):
                    continue
                
                #relations.pop(combination)
                typex_id = 1 #project.RELATIONS.index(a2_annotation.typex) + 1 
                element1 = entities[combination[0]]
                element2 = entities[combination[1]]
                idx1 = a2_annotation.element1_idx
                idx2 = a2_annotation.element2_idx
                a2_annotation.element1_idx = entities[idx1].name
                a2_annotation.element2_idx = entities[idx2].name
                if s.has_key(ind):
                    a = s[ind]
                    a.append(a2_annotation)
                    s[ind] = a
                else:
                    s[ind] = [a2_annotation]
            else: # negative examples
                typex_id = 0
                element1 = entities[combination[0]]
                element2 = entities[combination[1]]
                if (element1.typex not in project.Composes_Primary_Structure['DNA_Part'] or \
                   element2.typex not in project.Composes_Primary_Structure['DNA']) and \
                  (element1.typex not in project.Composes_Protein_Complex['Amino_Acid_Sequence'] or \
                   element2.typex not in project.Composes_Protein_Complex['Protein_Complex']) and\
                   (element1.typex not in project.Is_Member_Of_Family['Element'] or \
                   element2.typex not in project.Is_Member_Of_Family['Family']) and \
                   (element1.typex not in project.Is_Protein_Domain_Of['Domain'] or \
                   element2.typex not in project.Is_Protein_Domain_Of['Product']):
                    continue
            #entity_type_1 = element1.typex
            #entity_type_2 = element2.typex
            index = get_index(element1,element2,sentence)
            contexts = get_context(index[0],index[1],index[2],index[3],sentence,window)
            contexts = [str(word.word) for word in contexts]
            if len(contexts)==0:
                continue
            output_file.write(str(typex_id)+' '+ ' '.join(contexts)+'\n')
        ind += 1
        if len(s)==0:
            continue
        words = [w.word for w in sentence]
        o.write(str(a_i)+'. '+' '.join(words)+'\n')
        a_i += 1
        for k,v in s.iteritems():
            for a2 in v:
                o.write('  '+str(a2)+'\n')
        for idx in sentence_entity_set:
            o.write('  '+str(entities[idx])+'\n')
    o.close()
    

def close_file(*f):
    for fl in f:
        fl.close()
        
if __name__ == '__main__':
    train = open('train','w')
    dev = open('dev','w')
    window = 2
    for fn in glob.glob(path.GT_PROCESS_TRAIN+'/*.gt'):
        #print fn
        pmid = fn.split('\\')[-1][12:-3]
        gt_file = open(fn)
        a1_file = open(path.SOURCE_DATA_BINARY_TRAIN+'/SeeDev-binary-'+pmid+'.a1')
        a2_file = open(path.SOURCE_DATA_BINARY_TRAIN+'/SeeDev-binary-'+pmid+'.a2')
        gdep_file = open(path.GDEP_PROCESS_TRAIN+'/SeeDev-full-'+pmid+'.txt.gdep')
        # for linear context        
        create_examples(gt_file,a1_file,a2_file,train,window)
        #create_shortest_path_examples(gt_file,a1_file,a2_file,gdep_file,train)
        close_file(a1_file,a2_file,gt_file,gdep_file)
        
    for fn in glob.glob(path.GT_PROCESS_DEV+'/*.gt'):
        pmid = fn.split('\\')[-1][12:-3]
        gt_file = open(fn)
        a1_file = open(path.SOURCE_DATA_BINARY_DEV+'/SeeDev-binary-'+pmid+'.a1')
        a2_file = open(path.SOURCE_DATA_BINARY_DEV+'/SeeDev-binary-'+pmid+'.a2')
        gdep_file = open(path.GDEP_PROCESS_DEV+'/SeeDev-full-'+pmid+'.txt.gdep')
        create_examples(gt_file,a1_file,a2_file,dev,window)
        #create_shortest_path_examples(gt_file,a1_file,a2_file,gdep_file,dev)
        close_file(a1_file,a2_file,gt_file,gdep_file)
        #print fn
    dev.close()
    train.close()
