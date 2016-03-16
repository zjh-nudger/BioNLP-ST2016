#coding=gbk
"""
Created on Fri Apr 17 10:23:19 2015

@author: zjh
"""
import numpy,theano
import theano.tensor as T

'''
dataset='train.txt','valid.txt','test.txt'
'''
def load_bionlp_data(dataset,normalization_sigmoid=False,normalization_tanh=False):
    train_set=dataset[0]
    valid_set=dataset[1]
    test_set=dataset[2]
    norm_sigmoid=normalization_sigmoid
    norm_tanh=normalization_tanh
    #print 'normlization_tanh:'+str(normalization_tanh)
    def normalization_sigmoid(matrix,max=None,min=None):
        #print 'normalization_sigmoid'
    #    print np.shape(matrix)
        if max==None:
            max=matrix.max(axis=0) #取每一列的最大最小值
        if min==None:
            min=matrix.min(axis=0)
        matrix=-0.0+((matrix-min) /(max-min))*(1.0-0.0)
        #write_matrix_to_file(matrix=matrix,output='matrix.txt')
        return matrix
    def normalization_tanh(matrix,max=None,min=None):
    #    print np.shape(matrix)
        if max==None:
            max=matrix.max(axis=0) #取每一列的最大最小值
        if min==None:
            min=matrix.min(axis=0)
        matrix=-1.0+((matrix-min) /(max-min))*(1.0+1.0)
        return matrix
        
    def build_data_matrix(data_file):
        row=0
        for count,line in enumerate(open(data_file)):
            row+=1                   
        for line in open(data_file):
            col=len(line.split())-1
            break
        label=numpy.zeros((row,))
        data=numpy.zeros((row,col))        
        f=open(data_file,'r')
        row=0
        for line in f:
            eachline=line.split()
            label[row]=int(float(eachline[0]))
            col=0        
            for item in eachline[1:]:
                data[row][col]=float(item)
                col+=1
            row+=1
        f.close()
        
        label=label.reshape((row,1))
#        print numpy.shape(data),numpy.shape(label),row
        tmp=numpy.hstack((label,data))# shuffle the data
        numpy.random.permutation(tmp)
        label=tmp[:,:1]
        label=label.reshape((row,))
        data=tmp[:,1:]
        #print numpy.shape(data),numpy.shape(label)
        
        return data,label
        
    train_set=build_data_matrix(train_set)
    if valid_set!=None:
        valid_set=build_data_matrix(valid_set)
    test_set=build_data_matrix(test_set)
    
    
    def shared_dataset(data_xy, borrow=True):
        """
        将数据集放到共享变量里，每次将一个MiniBatch数据复制到GPU
        """
        data_x, data_y = data_xy
        
        if norm_sigmoid:
            data_x=normalization_sigmoid(data_x)
        if norm_tanh:
            data_x=normalization_tanh(data_x)
        
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)# label as float
        return shared_x, T.cast(shared_y, 'int32')

    #print str(normalization_sigmoid),str(normalization_tanh)    
    test_set_x, test_set_y = shared_dataset(test_set)
    if valid_set!=None:
        valid_set_x,valid_set_y=shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)
    if valid_set!=None:
        rval=[(train_set_x, train_set_y),(valid_set_x,valid_set_y),(test_set_x, test_set_y)]
    else:
        rval = [(train_set_x, train_set_y),None,(test_set_x, test_set_y)]
    return rval



def normalization_sigmoid(matrix,max=None,min=None):
    #print 'normalization_sigmoid'
#    print np.shape(matrix)
    if max==None:
        max=matrix.max(axis=0) #取每一列的最大最小值
    if min==None:
        min=matrix.min(axis=0)
    matrix=-0.0+((matrix-min) /(max-min))*(1.0-0.0)
    #write_matrix_to_file(matrix=matrix,output='matrix.txt')
    return matrix
def normalization_tanh(matrix,max=None,min=None):
#    print np.shape(matrix)
    if max==None:
        max=matrix.max(axis=0) #取每一列的最大最小值
    if min==None:
        min=matrix.min(axis=0)
    matrix=-1.0+((matrix-min) /(max-min))*(1.0+1.0)
    return matrix

'''
将一个矩阵写出到文件中..
'''
def write_matrix_to_file(matrix=None,output='none.txt'):
    output_file=open(output,'w')
    for instance in matrix.tolist():
        i=[str(j) for j in instance]
        output_file.write(' '.join(i)) 
        output_file.write('\n')
        #print '\n'
    output_file.close()
    

##without label
def build_data_matrix(data_file):
    row=0
    for count,line in enumerate(open(data_file)):
        row+=1                   
    for line in open(data_file):
        col=len(line.split())
        break
    data=numpy.zeros((row,col))        
    f=open(data_file,'r')
    row=0
    for line in f:
        eachline=line.split()
        col=0        
        for item in eachline:
            data[row][col]=float(item)
            col+=1
        row+=1
    f.close()
    
    return data

if __name__=='__main__':
    data='../test.txt'