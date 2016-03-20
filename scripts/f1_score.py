# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 16:24:56 2016

@author: tanfan.zjh
"""
import glob

tp = 0
all_positive = 0
predict_positive = 0

for fn in glob.glob('a2_type/*'):
    file_name = fn.split('\\')[-1]
    predict_file = open(fn)
    gold_file = open('../data/SeeDev/binary_dev/'+file_name)
    gold = {}
    for line in gold_file:
        toks = line.strip().split('\t')
        event_name = toks[1].strip().split()[0]
        e1_idx = toks[1].strip().split()[1].split(':')[1].strip()
        e2_idx = toks[1].strip().split()[2].split(':')[1].strip()
        
        if event_name == 'Has_Sequence_Identical_To' or \
            event_name == 'Is_Functionally_Equivalent_To':
                event_name = 'Is_Linked_To'
                
        if gold.has_key((e1_idx,e2_idx)):
            l = gold[(e1_idx,e2_idx)]
            l.append(event_name)
            all_positive += 1
        else:
            gold[(e1_idx,e2_idx)] = [event_name]
            all_positive += 1
    gold_file.close()
    for line in predict_file:
        if len(line) < 2:
            continue
        predict_positive += 1
        toks = line.strip().split('\t')
        event_name = toks[1].strip().split()[0]
        e1_idx = toks[1].strip().split()[1].split(':')[1].strip()
        e2_idx = toks[1].strip().split()[2].split(':')[1].strip()
        if gold.has_key((e1_idx,e2_idx)) and event_name in gold[(e1_idx,e2_idx)]:
            tp += 1
    predict_file.close()

precision = float(tp) / predict_positive
recall = float(tp) / all_positive
f1_score = 2 * precision*recall / (precision + recall)
print f1_score