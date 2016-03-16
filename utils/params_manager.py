#coding=gbk
"""
Created on Fri Apr 17 10:35:23 2015

@author: zjh
"""
import os,theano,cPickle,numpy

def load_params(param_dir='/home/liuxiaoming/deep_learning',\
                layer_num=3,
                prefix='/saved_params_layer_'):
    params={} #层数对应参数 0:(w,b)
    #print 'prefix:%s'%prefix
    for i in xrange(layer_num):
        file_name=param_dir+'/'+prefix%i
        #print param_dir
       # print file_name
        locals()['W_layer_%s'%i]=None
        locals()['b_layer_%s'%i]=None
        if os.path.exists(file_name):
            try:
                locals()['W_layer_%s'%i]=theano.shared(value=numpy.zeros((1,1),dtype=theano.config.floatX),\
                                     name='W_layer_%s'%i,borrow=True)
                locals()['b_layer_%s'%i]=theano.shared(value=numpy.zeros((1,),dtype=theano.config.floatX),\
                                     name='b_layer_%s'%i,borrow=True)
                files=open(file_name,'r')
                locals()['W_layer_%s'%i].set_value(cPickle.load(files),borrow=True)
                locals()['b_layer_%s'%i].set_value(cPickle.load(files),borrow=True)
                files.close()
            except EOFError:
                locals()['W_layer_%s'%i]=None
                locals()['b_layer_%s'%i]=None

        params.setdefault(i,(locals()['W_layer_%s'%i],locals()['b_layer_%s'%i]))
    return params

#将参数输出至文件
def output_params(params,params_path,params_prefix,layer):
    import os
    if not os.path.exists(params_path):
        os.mkdir(params_path)
    output_file=open(params_path+'/__layer%d__'%layer+params_prefix,'wb')
    cPickle.dump(params[0].get_value(borrow=True),output_file,-1)
    cPickle.dump(params[1].get_value(borrow=True),output_file,-1)
    output_file.close()
    