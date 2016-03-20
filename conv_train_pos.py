# -*- coding: utf-8 -*-
"""
Created on Sun May 10 15:07:35 2015

add part of speech features

@author: zjh
"""

import theano,numpy,sys
sys.path.append('..')
import theano.tensor as T
from collections import OrderedDict
from module.mlp_zjh import MLPDropout
from module.convnet import LeNetConvPoolLayer
from utils import evaluate

def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)
def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)
def Tanh(x):
    y = T.tanh(x)
    return(y)
def Iden(x):
    y = x
    return(y)
import numpy as np

def write_matrix_to_file(matrix=None,output='none.txt'):
    output_file=open(output,'w')
    for instance in matrix.tolist():
        i=str(instance)
        output_file.write(i) 
        output_file.write('\n')
        #print '\n'
    output_file.close()

claz_count = 21
def train_conv(datasets,
               wordvec,
               pos_emb,
               word_size=50,
               pos_size = 20,
               window_sizes=[9,11,13],
               hidden_units=[100,100,claz_count],
               dropout_rate=[0],
               shuffle_batch=True,
               n_epochs=10000,
               batch_size=256,
               lr_decay=0.95,
               sqr_norm_lim=9,
               conv_non_linear="relu",
               activations=[Tanh],#dropout
               non_static=True,
               proportion=1):
    rng = numpy.random.RandomState(3435)
    
    sen_length = len(datasets[0][0])-1  # sentence length
    filter_w = word_size + pos_size # filter width
    feature_maps = hidden_units[0]
    filter_shapes = [] #filter:param W
    pool_sizes = []
    for filter_h in window_sizes: # filter heighth
        filter_shapes.append((feature_maps, 1, filter_h,filter_w))
        pool_sizes.append((sen_length-filter_h+1, 1))
    parameters = [("image shape",sen_length,word_size),("filter shape",filter_shapes), ("hidden_units",hidden_units),
                  ("dropout", dropout_rate), ("batch_size",batch_size),("non_static", non_static),
                    ("learn_decay",lr_decay), ("conv_non_linear", conv_non_linear), ("non_static", non_static)
                    ,("sqr_norm_lim",sqr_norm_lim),("shuffle_batch",shuffle_batch)]
    print parameters  

    #print wordvec
    #count = np.shape(wordvec)[0]
    #wordvec=np.random.uniform(-0.25,0.25,(count,10))
    #wordvec=numpy.asarray(wordvec,dtype=theano.config.floatX)
    Words=theano.shared(value=wordvec,name='Words')
    POS = theano.shared(value=as_floatX(pos_emb),name='POS')    
    zero_vec_tensor = T.vector()
    zero_vec_tensor_pos = T.vector()
    zero_vec = numpy.zeros(word_size,dtype=theano.config.floatX)
    zero_vec_pos = numpy.zeros(pos_size,dtype=theano.config.floatX)
    set_zero = theano.function([zero_vec_tensor], 
                               updates=[(Words, T.set_subtensor(Words[0,:], zero_vec_tensor))])
    set_zero_pos = theano.function([zero_vec_tensor_pos], 
                               updates=[(POS, T.set_subtensor(POS[0,:], zero_vec_tensor_pos))])
    x=T.matrix('x')
    x_pos = T.matrix('x_pos')
    y=T.ivector('y')
    index=T.lscalar('index')
    
    layer0_input_word = Words[T.cast(x.flatten(),dtype="int32")].\
                        reshape((x.shape[0],1,x.shape[1],Words.shape[1]))
    pos_input = POS[T.cast(x_pos.flatten(),dtype="int32")].\
                    reshape((x_pos.shape[0],1,x_pos.shape[1],POS.shape[1]))
    layer0_input = T.concatenate([layer0_input_word,pos_input],axis=3)
    layer0_input = T.cast(layer0_input,dtype='float32')
    #theano.printing.debugprint(layer0_input)
    conv_layers=[]
    layer1_inputs=[]
    for i in xrange(len(window_sizes)):
        filter_shape = filter_shapes[i]
        pool_size = pool_sizes[i]
        conv_layer = LeNetConvPoolLayer(rng, input=layer0_input,image_shape=(batch_size, 1, sen_length, word_size + pos_size),
                                filter_shape=filter_shape, poolsize=pool_size, non_linear=conv_non_linear)
        layer1_input = conv_layer.output.flatten(2)
        conv_layers.append(conv_layer)
        layer1_inputs.append(layer1_input)
    layer1_input = T.concatenate(layer1_inputs,1)
    hidden_units[0] = feature_maps*len(window_sizes)
    #print hidden_units
    classifier = MLPDropout(rng, 
                            input=layer1_input, 
                            layer_sizes=hidden_units,
                            activations=activations, 
                            dropout_rates=dropout_rate)
    params = classifier.params   
    for conv_layer in conv_layers:
        params += conv_layer.params
    if non_static:
        params += [Words]
        params += [POS]

    cost = classifier.negative_log_likelihood(y) 
    dropout_cost = classifier.dropout_negative_log_likelihood(y)           
    grad_updates = sgd_updates_adadelta(params, dropout_cost, 
                                        lr_decay, 1e-6, sqr_norm_lim)
    
    numpy.random.seed(3435)

    test_set = datasets[2]
    train_set = datasets[0]
    train_pos_set = datasets[1]
    test_pos_set = datasets[3]
        
    if train_set.shape[0] % batch_size > 0:
        extra_data_num = batch_size - train_set.shape[0] % batch_size
        all_data = numpy.random.permutation(numpy.append(train_set,train_pos_set,axis=1))
        p_all = numpy.random.permutation(all_data) ## shuffle 
        train_set = p_all[:,:np.shape(train_set)[1]]
        train_pos_set = p_all[:,np.shape(train_set)[1]:]
        extra_data = train_set[:extra_data_num] # the batch
        extra_data_pos = train_pos_set[:extra_data_num]
        new_data=numpy.append(train_set,extra_data,axis=0) #使得训练集个数正好是batch_size的整数倍
        new_data_pos = numpy.append(train_pos_set,extra_data_pos,axis=0)
    else:
        new_data = train_set
        new_data_pos = train_pos_set
    #train_set = numpy.random.permutation(new_data)
    train_set_x = new_data[:,1:]
    test_set_x = test_set[:,1:]
    train_set_x, train_set_y = shared_dataset((train_set_x,new_data[:,0]))
    test_set_x, test_set_y = shared_dataset((test_set_x,test_set[:,0]))
    train_pos_set_sh = shared_dataset_x(new_data_pos)
    test_pos_set = shared_dataset_x(test_pos_set)
    n_batches = new_data.shape[0]/batch_size #batch num
    n_train_batches = int(numpy.round(n_batches))
    
    train_model = theano.function([index], cost, updates=grad_updates,
          givens={
            x: train_set_x[index*batch_size:(index+1)*batch_size],
            y: train_set_y[index*batch_size:(index+1)*batch_size],
            x_pos : train_pos_set_sh[index*batch_size:(index+1)*batch_size]})
    #theano.printing.debugprint(train_model)


    test_pred_layers = []
    test_size = test_set_x.shape[0].eval()
    test_layer0_input_1 = Words[T.cast(x.flatten(),dtype="int32")].\
                              reshape((x.shape[0],1,x.shape[1],Words.shape[1]))
    pos_input = POS[T.cast(x_pos.flatten(),dtype="int32")].\
                    reshape((x_pos.shape[0],1,x_pos.shape[1],POS.shape[1]))
    test_layer0_input = T.concatenate([test_layer0_input_1,pos_input],axis=3)
    test_layer0_input = T.cast(test_layer0_input,dtype='float32')
    for conv_layer in conv_layers:
        test_layer0_output = conv_layer.predict(test_layer0_input, test_size)
        test_pred_layers.append(test_layer0_output.flatten(2))
    test_layer1_input = T.concatenate(test_pred_layers, 1)
    test_y_pred = classifier.predict(test_layer1_input)
    test_error = T.mean(T.neq(test_y_pred, y))
    test_model_all = theano.function(inputs=[x,x_pos,y], outputs=[test_error,test_y_pred])   
       
    epoch=0
    max_f1_score = 0.17
    while (epoch < n_epochs):
        epoch+=1        
        if shuffle_batch:
            cost=[]
            for minibatch_index in numpy.random.permutation(range(n_train_batches)):
                cost_epoch = train_model(minibatch_index)
                cost.append(cost_epoch)
                set_zero(zero_vec)
                set_zero_pos(zero_vec_pos)
            error,prediction=test_model_all(x=test_set_x.get_value(borrow=True),\
                                            x_pos=test_pos_set.get_value(borrow=True),\
                                            y=test_set_y.eval())
            precision,recall,f1_score=evaluate.evaluate_multi_class_seedev(prediction=prediction,
                                                                          answer=test_set_y.eval(),
                                                                          claz_count=claz_count)
            #print 'epoch:%d,error:%.3f,micro_f_score:%.2f,macro_f_score:%.2f'%(epoch,error,micro_f_score,macro_f_score)
            print 'epoch:%d,error:%.3f,precision:%.4f,  recall:%.4f,  f1_score:%.4f'%(epoch,error,precision,recall,f1_score)
            if f1_score > max_f1_score:
                max_f1_score = f1_score
                write_matrix_to_file(prediction,'pred_21.txt')
        else:
            for minibatch_index in xrange(n_train_batches):
                cost_epoch = train_model(minibatch_index)
                set_zero(zero_vec)
#            error,prediction=test_model_all(x=test_set_x.get_value(borrow=True),y=test_set_y.eval())
#            micro_f_score,macro_f_score=evaluate.simple_evaluate(prediction=prediction,
#                                                                 answer=test_set_y.eval())
            print 'epoch:%d,error:%f'%(epoch,error)
#    write_matrix_to_file(prediction,'pred.txt')
#    write_matrix_to_file(test_set_y.eval(),'real.txt')
        
        
def shared_dataset_x(data_x, borrow=True):
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x


def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

def as_floatX(variable):
    if isinstance(variable, float):
        return numpy.cast[theano.config.floatX](variable)

    if isinstance(variable, numpy.ndarray):
        return numpy.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)        
        
def sgd_updates_adadelta(params,cost,rho=0.95,epsilon=1e-6,norm_lim=9,word_vec_name='Words'):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = numpy.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty),name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gp = T.cast(gp,dtype='float32')        
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = T.cast(up_exp_sg,dtype='float32')
        step =  -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = T.cast(rho * exp_su + (1 - rho) * T.sqr(step),dtype='float32')
        stepped_param = param + step
        if (param.get_value(borrow=True).ndim == 2) and (param.name!='Words') and (param.name!='POS'):
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            tmp=stepped_param * scale
            tmp=T.cast(tmp,'float32')
            #print param.type,tmp.type
            updates[param] = tmp
        else:
            updates[param] = T.cast(stepped_param,dtype = 'float32')
            #print param.type,stepped_param.type
    return updates


if __name__=='__main__':
    import cPickle as cp
    theano.config.floatX='float32'
    train,train_pos,test,test_pos,words,pos_emb=cp.load(open('data/data_entity_type.p'))
    words=numpy.asarray(words,dtype=theano.config.floatX)
    train_conv(datasets=[train,train_pos,test,test_pos],wordvec=words,pos_emb=pos_emb)
