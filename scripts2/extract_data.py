# -*- coding: utf-8 -*- 
"""
Created on Sat Mar 12 19:16:19 2016

@author: tanfan.zjh
"""
import xlrd,re
import codecs


## 2 title 4 abstrct 7 topic 9 keywords 40 语言 48 ISBN/ISSN 49 索取号
data = xlrd.open_workbook('ab.xls')
table = data.sheets()[0]
index = 0

idx_set = set()
output = codecs.open('unlabeled_data/ab.txt','w',encoding='utf-8')
idx_file = open('unlabeled_data/idx.txt','w')
for row in xrange(table.nrows-1):
    row += 1
    language = table.cell(row,40).value
    if not (language == 'eng' or language == 'ENG'):
        continue
    title = table.cell(row,2).value
    abstract = table.cell(row,4).value
    topic = table.cell(row,7).value
    keywords = table.cell(row,9).value
    idx = table.cell(row,49).value
    if idx in idx_set:
        continue
    else:
        idx_set.add(idx)
        idx_file.write(str(idx)+'\n')
    #output_dir = 'unlabeled_data'
    output.write(re.sub('\r|\n',' ',title)+'\n')
    #output.write(re.sub('\r|\n',' ',keywords)+'\n')
    output.write(re.sub('\r|\n',' ',abstract)+'\n')
output.close()
idx_file.close()
    
    
    