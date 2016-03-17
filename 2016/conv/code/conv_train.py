# -*- coding: utf-8 -*-
"""
"""
import theano,numpy,sys
sys.path.append('..')
import theano.tensor as T
from collections import OrderedDict
from module.mlp import MLPDropout
from module.convnet import LeNetConvPoolLayer
import csv
from utils import evaluate
from numpy import asarray

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

def train_conv(datasets,
               wordvec,
               word_size=200,
               window_sizes=[11,13,15],
               hidden_units=[100,200,23],
               dropout_rate=[0],
               shuffle_batch=True,
               n_epochs=1000,
               batch_size=128,
               lr_decay=0.95,
               sqr_norm_lim=9,
               conv_non_linear="relu",
               activations=[Tanh],#dropout
               non_static=True,
               proportion=1):
    rng = numpy.random.RandomState(3435)
    
    sen_length = len(datasets[0][0])-1  # sentence length
    filter_w = word_size   # filter width
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
    Words=theano.shared(value=wordvec,name='Words')
    zero_vec_tensor = T.vector()
    zero_vec = numpy.zeros(word_size,dtype=theano.config.floatX)
    set_zero = theano.function([zero_vec_tensor], 
                               updates=[(Words, T.set_subtensor(Words[0,:], zero_vec_tensor))])
    x=T.matrix('x')
    y=T.ivector('y')
    index=T.lscalar('index')
    
    layer0_input = Words[T.cast(x.flatten(),dtype="int32")].\
                        reshape((x.shape[0],1,x.shape[1],Words.shape[1]))
    #theano.printing.debugprint(layer0_input)
    conv_layers=[]
    layer1_inputs=[]
    for i in xrange(len(window_sizes)):
        filter_shape = filter_shapes[i]
        pool_size = pool_sizes[i]
        conv_layer = LeNetConvPoolLayer(rng, input=layer0_input,image_shape=(batch_size, 1, sen_length, word_size),
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
        params+=[Words]

    cost = classifier.negative_log_likelihood(y) 
    dropout_cost = classifier.dropout_negative_log_likelihood(y)           
    grad_updates = sgd_updates_adadelta(params, dropout_cost, 
                                        lr_decay, 1e-6, sqr_norm_lim)
    
    numpy.random.seed(3435)

    test_set = datasets[1]
    train_set = datasets[0]
        
    if train_set.shape[0] % batch_size > 0:
        extra_data_num = batch_size - train_set.shape[0] % batch_size
        train_set = numpy.random.permutation(train_set) ## shuffle  
        extra_data = train_set[:extra_data_num] # the batch
        new_data=numpy.append(train_set,extra_data,axis=0) #使得训练集个数正好是batch_size的整数倍
    else:
        new_data = train_set
    train_set = numpy.random.permutation(new_data)
    train_set_x = train_set[:,1:]
    test_set_x = test_set[:,1:]
    train_set_x, train_set_y = shared_dataset((train_set_x,train_set[:,0]))
    test_set_x, test_set_y = shared_dataset((test_set_x,test_set[:,0]))

    n_batches = new_data.shape[0]/batch_size #batch num
    n_train_batches = int(numpy.round(n_batches))

    train_model = theano.function([index], cost, updates=grad_updates,
          givens={
            x: train_set_x[index*batch_size:(index+1)*batch_size],
            y: train_set_y[index*batch_size:(index+1)*batch_size]})   
    #theano.printing.debugprint(train_model)
    
    test_pred_layers = []
    test_size = test_set_x.shape[0].eval()
    test_layer0_input = Words[T.cast(x.flatten(),dtype="int32")].\
                              reshape((x.shape[0],1,x.shape[1],Words.shape[1]))
    for conv_layer in conv_layers:
        test_layer0_output = conv_layer.predict(test_layer0_input, test_size)
        test_pred_layers.append(test_layer0_output.flatten(2))
    test_layer1_input = T.concatenate(test_pred_layers, 1)
    test_y_pred = classifier.predict(test_layer1_input)
    test_error = T.mean(T.neq(test_y_pred, y))
    test_model_all = theano.function(inputs=[x,y], outputs=[test_error,test_y_pred])   

    macro=[]
    micro=[]
    epoch=0
    #li-输出
    trigger_answer=test_set_y.eval().tolist()
    answer_list=[]
    for line in trigger_answer:
        answer_list.append(int(line))
    Li_out = []  
    Li_out.append(answer_list)
    #li-输出
    while (epoch < n_epochs):
        epoch+=1        
        if shuffle_batch:
            cost=[]
            for minibatch_index in numpy.random.permutation(range(n_train_batches)):
                cost_epoch = train_model(minibatch_index)
                cost.append(cost_epoch)
                set_zero(zero_vec)
            error,prediction=test_model_all(x=test_set_x.get_value(borrow=True),y=test_set_y.eval())
              
            micro_f_score,macro_f_score=evaluate.simple_evaluate(prediction=prediction,
                                                                 answer=test_set_y.eval(),out=Li_out)
            print 'epoch:%d,error:%f,micro_f_score:%f,macro_f_score:%f'\
                    %(epoch,error,micro_f_score,macro_f_score)
            macro.append(macro_f_score)
            micro.append(micro_f_score)
            
        else:
            for minibatch_index in xrange(n_train_batches):
                cost_epoch = train_model(minibatch_index)
                set_zero(zero_vec)
            error,prediction=test_model_all(x=test_set_x.get_value(borrow=True),y=test_set_y.eval())
            micro_f_score,macro_f_score=evaluate.simple_evaluate(prediction=prediction,
                                                                 answer=test_set_y.eval())
            print 'epoch:%d,error:%f,micro_f_score:%f,macro_f_score:%f'\
                    %(epoch,error,micro_f_score,macro_f_score)
            macro.append(macro_f_score)
            micro.append(micro_f_score)

    print 'max micro value:%f'%(numpy.max(micro))
    print 'max macro value:%f'%(numpy.max(macro))
    #li-输出
    csv_writer=csv.writer(open('/home/lihaorui/2016/li-out-3.csv','wb'))
    Li_out = asarray(Li_out)
    for i in range(len(Li_out)):
        csv_writer.writerow(Li_out[i])
    #li-输出
    evaluate.complete_evaluate(prediction,test_set_y.eval(),
                       csv_output='/home/lihaorui/2016/data/out/3.csv')
    macro=[str(i) for i in macro]
    micro=[str(i) for i in micro]
    record_macro_file=open('/home/lihaorui/2016/data/out/macro_3.txt','a')
    record_macro_file.write(' '.join(macro)+'\n')
    record_macro_file.close()
    record_macro_file=open('/home/lihaorui/2016/data/out/micro_3.txt','a')
    record_macro_file.write(' '.join(micro)+'\n')
    record_macro_file.close()
        

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
            tmp=T.cast(tmp,'float32')
            #print param.type,tmp.type
            updates[param] = tmp
        else:
            updates[param] = stepped_param
            #print param.type,stepped_param.type
    return updates 


if __name__=='__main__':
    import cPickle as cp
    theano.config.floatX='float32'
    train,test,words=cp.load(open('./data/out/Li.p'))
    words=numpy.asarray(words,dtype=theano.config.floatX)
    train_conv(datasets=[train,test],wordvec=words)