# -*- coding: utf-8 -*-
"""
Created on Sun May 10 15:07:35 2015

@author: zjh
"""

import theano,numpy,sys
sys.path.append('..')
import theano.tensor as T
from collections import OrderedDict
from module.mlp_zjh import MLPDropout
<<<<<<< HEAD
import numpy as np
=======
>>>>>>> fcaa5859445351e40525e8d9eee56180be4b5d04

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

def train_nn(datasets,
             wordvec,
<<<<<<< HEAD
             word_size=100,
             hidden_units=[800,800,6],
             dropout_rate=[0,0,0,0,0,0,0],
=======
             word_size=200,
             hidden_units=[2000,1000,23],
             dropout_rate=[0],
>>>>>>> fcaa5859445351e40525e8d9eee56180be4b5d04
             shuffle_batch=True,
             n_epochs=3000,
             batch_size=256,
             init_learning_rate=0.4,
             adadelta=True,
             lr_decay=0.95,
             sqr_norm_lim=9,
<<<<<<< HEAD
             activations=[ReLU,ReLU,ReLU,ReLU,ReLU,ReLU,ReLU],
=======
             activations=[Tanh],
>>>>>>> fcaa5859445351e40525e8d9eee56180be4b5d04
             non_static=True,
             use_valid_set=False,
             proportion=1):
    rng = numpy.random.RandomState(3435)
<<<<<<< HEAD
    #print np.shape(wordvec)
    count = np.shape(wordvec)[0]
    wordvec=np.random.uniform(-0.25,0.25,(count,100))
    wordvec=numpy.asarray(wordvec,dtype=theano.config.floatX)
=======
    #print wordvec
>>>>>>> fcaa5859445351e40525e8d9eee56180be4b5d04
    Words=theano.shared(value=wordvec,name='Words')
    zero_vec_tensor = T.vector()
    zero_vec = numpy.zeros(word_size,dtype=theano.config.floatX)
    set_zero = theano.function([zero_vec_tensor], 
                               updates=[(Words, T.set_subtensor(Words[0,:], zero_vec_tensor))])
    x=T.matrix('x')
    y=T.ivector('y')
    index=T.lscalar('index')
    
    layer0_input= Words[T.cast(x.flatten(),dtype="int32")].\
                  reshape((x.shape[0],x.shape[1]*Words.shape[1]))
    #input_printed=theano.printing.Print('layer0_input:')(layer0_input)
    classifier = MLPDropout(rng, 
                            input=layer0_input, 
                            layer_sizes=hidden_units,
                            activations=activations, 
                            dropout_rates=dropout_rate)
    params = classifier.params
    
    if non_static:
        params.append(Words)
        
    cost = classifier.negative_log_likelihood(y) 
    dropout_cost = classifier.dropout_negative_log_likelihood(y)  
    if adadelta:
        grad_updates = sgd_updates_adadelta(params, dropout_cost, 
                                            lr_decay, 1e-6, sqr_norm_lim)
    else:
        grad_updates = sgd_updates(params,dropout_cost,init_learning_rate)
    #print params
    numpy.random.seed(3435)
    if datasets[0].shape[0] % batch_size > 0:
        extra_data_num = batch_size - datasets[0].shape[0] % batch_size
        train_set = numpy.random.permutation(datasets[0]) ## shuffle  
        extra_data = train_set[:extra_data_num] # the batch
        new_data=numpy.append(datasets[0],extra_data,axis=0) #使得训练集个数正好是batch_size的整数倍
    else:
        new_data = datasets[0]
    
    new_data = numpy.random.permutation(new_data) #train data
    n_batches = new_data.shape[0]/batch_size #batch num
    n_train_batches = int(numpy.round(n_batches*proportion))
    
    if len(datasets)==3:
        use_valid_set=True
        train_set = new_data
        val_set = datasets[1]
        train_set_x, train_set_y = shared_dataset((train_set[:,1:],train_set[:,0]))
        val_set_x, val_set_y = shared_dataset((val_set[:,1:],val_set[:,0]))
        test_set_x = datasets[2][:,1:] 
        test_set_y = numpy.asarray(datasets[2][:,0],"int32")
    else:
        test_set_x = datasets[1][:,1:] 
        test_set_y = numpy.asarray(datasets[1][:,0],"int32")
        if use_valid_set:
            train_set = new_data[:n_train_batches*batch_size,:]
            val_set = new_data[n_train_batches*batch_size:,:]     
            train_set_x, train_set_y = shared_dataset((train_set[:,1:],train_set[:,0]))
            val_set_x, val_set_y = shared_dataset((val_set[:,1:],val_set[:,0]))
        else:
            train_set = new_data[:,:]
            train_set_x, train_set_y = shared_dataset((train_set[:,1:],train_set[:,0]))

    n_batches = new_data.shape[0]/batch_size #batch num
    n_train_batches = int(numpy.round(n_batches))

    train_model = theano.function([index], cost, updates=grad_updates,
          givens={
            x: train_set_x[index*batch_size:(index+1)*batch_size],
            y: train_set_y[index*batch_size:(index+1)*batch_size]})   
    #theano.printing.debugprint(train_model)
            
    #f_scores=[classifier.f_score(y,i+1) for i in xrange(hidden_units[-1]-1)]
<<<<<<< HEAD
    f_scores =(classifier.f_score(y,i+1)[0] for i in xrange(5))
=======
    f_scores =(classifier.f_score(y,i+1)[0] for i in xrange(22))
>>>>>>> fcaa5859445351e40525e8d9eee56180be4b5d04
    f_scores = tuple(f_scores)
    '''    
    fenzi=0
    fenmu=0    
    for item in f_scores:
        f_score,precision,recall=item
        fenzi+=(precision*recall)
        fenmu+=(precision+recall)
    '''
    #fenzi_printed=theano.printing.Print('fenzi:')(fenzi)
    #micro_avg_f_score=2.*fenzi / (fenmu+0.000001)
    test_model=theano.function([],f_scores,
                               givens={
                               x:test_set_x,
                               y:test_set_y})
    '''
    valid_model=theano.function([],classifier.errors(y),
                                givens={
                                x:test_set_x,
                                y:test_set_y})
    '''
    #theano.printing.debugprint(test_model)
    #print micro_avg_f_score.owner.inputs
    #test_y_pred = classifier.f_score()
    #test_model_all = theano.function([x,y], test_error)   

    epoch=0
    while (epoch < n_epochs):        
        epoch = epoch + 1
        if shuffle_batch:
            cost=[]
            for minibatch_index in numpy.random.permutation(range(n_train_batches)):
                cost_epoch = train_model(minibatch_index)
                cost.append(cost_epoch)
                set_zero(zero_vec)
            print 'epoch:%d, cost value:%f, '%(epoch,numpy.mean(cost)),
        else:
            for minibatch_index in xrange(n_train_batches):
                cost_epoch = train_model(minibatch_index)
                print cost_epoch
                set_zero(zero_vec)
        #print test_model()
        f_scores = test_model()
        f_scores = tuple(f_scores)
<<<<<<< HEAD
        print '%.2f,'*5%(f_scores)
=======
        print '%.2f,'*22%(f_scores)
>>>>>>> fcaa5859445351e40525e8d9eee56180be4b5d04
        
    layer0_input= Words[T.cast(test_set_x.flatten(),dtype="int32")].\
                  reshape((test_set_x.shape[0],test_set_x.shape[1]*Words.shape[1]))
    t_pred=classifier.predict(layer0_input)
    write_matrix_to_file(t_pred.eval(),'pred.txt')
    write_matrix_to_file(test_set_y,'real.txt')
    #print Words.get_value()

def shared_dataset(data_xy, borrow=True):

        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
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

def sgd_updates(params,cost,learning_rate):
    gparams=T.grad(cost,params)
    updates=[(param,param-learning_rate*gparam)
             for param,gparam in zip(params,gparams)]
    return updates
 
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
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step =  -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        if (param.get_value(borrow=True).ndim == 2) and (param.name!='Words'):
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            tmp=stepped_param * scale
            tmp=T.cast(tmp,theano.config.floatX)
            #print param.type,tmp.type
            updates[param] = tmp
        else:
            updates[param] = stepped_param
            #print param.type,stepped_param.type
    return updates 

def write_matrix_to_file(matrix=None,output='none.txt'):
    output_file=open(output,'w')
    for instance in matrix.tolist():
        i=str(instance)
        output_file.write(i) 
        output_file.write('\n')
        #print '\n'
    output_file.close()

if __name__=='__main__':
    import cPickle as cp
    theano.config.floatX='float32'
    train,test,words,vocab=cp.load(open('data/data.p'))
    words=numpy.asarray(words,dtype=theano.config.floatX)
    train_nn(datasets=[train,test],wordvec=words)