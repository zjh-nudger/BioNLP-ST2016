# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 11:03:41 2016

后处理

@author: tanfan.zjh
"""
import project

test_file = open('data/train_data/dev')
pred_file = open('data/pred_entity+disemb.txt')

instances = {}
for test,pred in zip(test_file,pred_file):
    pred_claz = int(pred.strip())
    idx = test.strip().split(' ')[0]
    if pred_claz > 0:
        instances[idx] = project.RELATIONS[pred_claz-1]

test_file.close()
pred_file.close()

event_list =  project.load_event_def()

import glob
for fn in glob.glob('data/a2_tmp_file/*.a2'):
    f = open(fn)
    file_name = fn.split('\\')[-1]
    a2_file = open('data/a2-dis/'+file_name,'w')
    index = 0
    for line in f:
        toks = line.strip().split('|')
        relation_idx = toks[0].strip()
        if not instances.has_key(relation_idx):
            continue
        e1 = toks[1].strip()
        toks_e1 = e1.split('\t')
        e1_idx = toks_e1[0].strip()
        e1_type = toks_e1[1].strip().split()[0].strip()
        e2 = toks[2].strip()
        toks_e2 = e2.split('\t')
        e2_idx = toks_e2[0].strip()
        e2_type = toks_e2[1].strip().split()[0].strip()
        relation_name = instances[relation_idx]
        event_def = event_list[relation_name]
        if event_def.e3_name == None:
            if (e1_type in event_def.e1_entity and e2_type in event_def.e2_entity):
                a2_file.write('E'+str(index)+'\t'+relation_name+' '+event_def.e1_name\
                             +':'+e1_idx+' '+event_def.e2_name+':'+e2_idx+'\n')
                index += 1
            elif (e2_type in event_def.e1_entity and e1_type in event_def.e2_entity):
                a2_file.write('E'+str(index)+'\t'+relation_name+' '+event_def.e1_name\
                             +':'+e2_idx+' '+event_def.e2_name+':'+e1_idx+'\n')
                index += 1
        else:
            if (e1_type in event_def.e1_entity and e2_type in event_def.e3_entity):
                a2_file.write('E'+str(index)+'\t'+relation_name+' '+event_def.e1_name\
                             +':'+e1_idx+' '+event_def.e3_name+':'+e2_idx+'\n')
                index += 1
            elif (e1_type in event_def.e2_entity and e2_type in event_def.e3_entity):
                a2_file.write('E'+str(index)+'\t'+relation_name+' '+event_def.e2_name\
                             +':'+e1_idx+' '+event_def.e3_name+':'+e2_idx+'\n')
                index += 1
            elif (e2_type in event_def.e1_entity and e1_type in event_def.e3_entity):
                a2_file.write('E'+str(index)+'\t'+relation_name+' '+event_def.e1_name\
                             +':'+e2_idx+' '+event_def.e3_name+':'+e1_idx+'\n')
                index += 1
            elif (e2_type in event_def.e2_entity and e1_type in event_def.e3_entity):
                a2_file.write('E'+str(index)+'\t'+relation_name+' '+event_def.e2_name\
                             +':'+e2_idx+' '+event_def.e3_name+':'+e1_idx+'\n')
                index += 1
        
    f.close()
    a2_file.close()
    