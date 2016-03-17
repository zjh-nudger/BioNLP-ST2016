# -*- coding: utf-8 -*-
"""
Created on Wed May 20 15:00:51 2015

@author: zjh
"""

import pylab

macro=open('micro_Around4_new1.txt')
#micro=open('result_micro.txt')
d={1:'non-static',2:'static'}
d={1:'adadelta',2:'momtenum'}
d={1:'one layer',2:'size=200',3:'pubmed size=200',4:'size=200 window=4',5:'size=200 window=6'}
d={1:'100',2:'150',3:'200',4:'300',5:'400'}
d={1:'pre',2:'no',3:'200',4:'300',5:'400'}
c=0
for item in macro:
    c=c+1
    items = item.split()
    x=[i+1 for i in xrange(len(items))]
    items_macro=[float(i) for i in items]
    pylab.plot(x,items_macro,linewidth=1.5, linestyle="-", label=d[c])

#micro.close()
macro.close()

pylab.title('macro..')

'''
micro=open('result_micro.txt')
for item in micro:
    items = item.split()
    
    items_micro=[float(i) for i in items]
micro.close()
'''



#pylab.plot(x,items_micro,linewidth=1.5, linestyle="-", label="micro F score")

pylab.legend(loc="lower right")

pylab.xlabel('epoch',fontproperties='SimHei')
pylab.ylabel(u'macro f score ',fontproperties='SimHei')




pylab.show()
#pylab.plot(x,y2)