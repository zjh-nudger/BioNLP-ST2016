# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 11:03:41 2016

后处理

@author: tanfan.zjh
"""
import project,glob

event_list =  project.load_event_def()

for fn in glob.glob('data/a2_predict/*'):
    file_name = fn.split('\\')[-1]
    f = open(fn)
    a2_file = open('data/a2_lhl/'+file_name,'w')
    index = 0
    for line in f:
        toks = line.strip().split()
        typex = int(toks[0])
        if typex == 0:
            continue
        relation_name = project.RELATIONS[typex-1]
        event_def = event_list[relation_name]
        e1_toks = toks[1].split('_')
        e1_idx = e1_toks[0]
        e1_type = '_'.join(e1_toks[1:])
        e2_toks = toks[2].split('_')
        e2_idx = e2_toks[0]
        e2_type = '_'.join(e2_toks[1:])
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
    a2_file.close()
    f.close()
