#coding:utf-8
"""
Created on Mon Mar 07 14:24:32 2016

@author: tanfan.zjh
"""

import path,glob
'''
for fn in glob.glob(path.SUPPORTING_RESOURCES_TOKENIZATION+'/train/SeeDev-full-9657152-1.txt'):
    ss_txt_file = open(fn)
    source_txt_file = open(path.SOURCE_DATA_BINARY_TRAIN+'/SeeDev-binary-9657152-1.txt')
    a1_file = open(path.SOURCE_DATA_BINARY_TRAIN+'/SeeDev-binary-9657152-1.a1')
    a2_file = open(path.SOURCE_DATA_BINARY_TRAIN+'/SeeDev-binary-9657152-1.a2')
'''
output_file = open('statics.txt','w')
relations = {}
import re
for fn in glob.glob(path.SOURCE_DATA_BINARY_TRAIN+'/*.a1'):
    file_name = fn.split('\\')[-1]
    entity_dic = {}
    a1_file = open(fn)
    a2_file = open(fn.replace('.a1','.a2'))
    for line in a1_file:
        toks = line.strip().split('\t')
        if ';' in toks[1]:
            toks1 = re.split(' |;',toks[1])
            if len(toks1)==5:
                start1 = int(toks1[1])
                end1 = int(toks1[2])
                start2 = int(toks1[3])
                end2 = int(toks1[4])
                entity_dic[toks[0]] = [start1,end1,start2,end2,toks1[0],toks[2]]
            if len(toks1)==7:
                start1 = int(toks1[1])
                end1 = int(toks1[2])
                start2 = int(toks1[3])
                end2 = int(toks1[4])
                start3 = int(toks1[5])
                end3 = int(toks1[6])
                entity_dic[toks[0]] = [start1,end1,start2,end2,start3,end3,toks1[0],toks[2]]
        else:
            start = int(toks[1].split()[1])
            end = int(toks[1].split()[2])
            entity_dic[toks[0]] = [start,end,toks[1].split()[0],toks[2]]
    for line in a2_file:
        toks = re.split('\t| |:',line.strip())
        assert len(toks) == 6
        relation_type = toks[1]
        element1_name = toks[2]
        element1 = entity_dic[toks[3]]
        element2_name = toks[4]
        element2 = entity_dic[toks[5]]
        if relations.has_key(relation_type):
            l = relations[relation_type]
            l.append(element1_name+':'+str(element1)+' '+\
                     element2_name+':'+str(element2)+' '+file_name)
        else:
            relations[relation_type] = [element1_name+':'+str(element1)+' '+\
                                        element2_name+':'+str(element2)+' '+file_name]
index = 0
for k,v in relations.iteritems():
    index += 1
    output_file.write('%d..%s #######################'%(index,k))
    output_file.write('\n')
    for e in v:
        output_file.write(str(e)+'\n')
    output_file.write('\n\n')
output_file.close()