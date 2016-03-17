#coding:gbk
"""
Created on Mon May 18 14:17:59 2015

@author: zjh
"""
import theano.tensor as T
import theano,numpy
from theano.ifelse import ifelse
from utils import params_manager as pm
from collections import OrderedDict
from module.sda import SDA
import cPickle as cp

def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)


vec_name='vecs_dep_all_50'
#vec_name='pubmed_w2v'
#vec_name='vecs_dep_all_10'
#vec_name='pubmed_my'

def sda_train(processed_data='data/data.p',#'../data/'+vec_name+'/data_around4_rand_vocab.p',
              finetune_lr=0.5, 
              training_epochs=500,
              batch_size=200,
              nn_structure=[1600,1000,20],
              dropout_rates=[0.5],
              L1_reg=0,L2_reg=0.0001,
              mom_params = {'start': 0.5,
                            'end': 0.99,
                            'interval': 500},
              params_path='/home/liuxiaoming/dl_zjh/paramsdddd',
              mom=False,
              adadelta=True,
              activation=ReLU,
              non_static=True):
    learning_rate = theano.shared(numpy.asarray(finetune_lr,
        dtype=theano.config.floatX))
    decay_learning_rate = theano.function(inputs=[], outputs=learning_rate,
            updates={learning_rate: learning_rate * 1})
    print processed_data
    squared_filter_length_limit = 15.0 # for momtumn config
    
    mypretrain_params=[('nn_structure',nn_structure),
                       ('corruption_levels',[0.])]

    myparams=[('nn_structure',str(nn_structure)),('finetune_lr',finetune_lr),
              ('batch_size',batch_size),('training_epochs',training_epochs),
              ('activation',str(activation)),('non_static',non_static),
              ('L2_reg',L2_reg),('L1_reg',L1_reg),('dropout_rate',dropout_rates)]
    print myparams 
    
    
    layer_size=len(nn_structure)-2
    params = pm.load_params(param_dir=params_path,prefix='__layer%d__'+str(mypretrain_params),layer_num=layer_size)
    for i in xrange(layer_size):
        if params[i][0]!=None:
            print '-----已有第%d层参数'%i
        else:
            if(i==0):
                print '-----没有已保存的参数...'
    processed_data=processed_data.strip()
    train,test,words,vocab=cp.load(open(processed_data))
    datasets=train,test

   # numpy_rng = numpy.random.RandomState(89677)


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

    index=T.lscalar('index') 
    #learning_rate = T.scalar('learning_rate')

    sda = SDA(
        W_sda=[params[i][0] for i in xrange(layer_size)],
        b_sda=[params[i][1] for i in xrange(layer_size)],
        nn_structure=nn_structure, 
        dropout_rates=dropout_rates,
        L1_reg=L1_reg,L2_reg=L2_reg,
        activation=activation,
        wordvec=words,
        non_static=non_static
    )
    

    batch_begin = index * batch_size
    batch_end = batch_begin + batch_size
    
    if not mom and not adadelta:
        gparams=T.grad(sda.finetune_cost,sda.params)
        updates=[
            (param, param - learning_rate * gparam)
            for param, gparam in zip(sda.params, gparams)
        ]
        updates=sgd_updates_adadelta(params=sda.params,cost=sda.finetune_cost)
        train_model = theano.function(inputs=[index],
                                      updates=updates,
                                      outputs=[sda.finetune_cost],
                                      givens={
                                          sda.x: train_set_x[batch_begin: batch_end],
                                          sda.y: train_set_y[batch_begin: batch_end]
                                      })
    
    if mom:
        epoch=T.scalar('epoch')
        updates=sgd_updates_mom(cost=sda.finetune_cost,
                                params=sda.params,
                                mom_param=mom_params,
                                epoch=epoch,
                                squared_filter_length_limit=squared_filter_length_limit,
                                learning_rate=learning_rate
                                )
        train_model = theano.function(inputs=[epoch,index],
                                      updates=updates,
                                      outputs=[sda.finetune_cost],
                                      givens={
                                          sda.x: train_set_x[batch_begin: batch_end],
                                          sda.y: train_set_y[batch_begin: batch_end]
                                      })
    if adadelta:
        updates=sgd_updates_adadelta(params=sda.params,cost=sda.finetune_cost)
        train_model = theano.function(inputs=[index],
                                      updates=updates,
                                      outputs=[sda.finetune_cost],
                                      givens={
                                          sda.x: train_set_x[batch_begin: batch_end],
                                          sda.y: train_set_y[batch_begin: batch_end]
                                      })
        
    for epoch in xrange(training_epochs):
        avg_cost=[]
        for mini_batch_index in xrange(n_train_batches):
            if mom:
                cost=train_model(epoch=epoch,index=mini_batch_index)
            else:
                cost=train_model(index=mini_batch_index)
            avg_cost.append(cost)
        prediction=sda.logLayer.y_pred.eval({sda.x:test_set_x.get_value(borrow=True)})
#        prediction_p=sda.logLayer.p_y_given_x.eval({sda.x:test_set_x.get_value(borrow=True)})
        error=sda.logLayer.errors(test_set_y).eval({sda.x:test_set_x.get_value(borrow=True)}) 
        print 'epoch:%d,error:%f'%(epoch,error)
        decay_learning_rate()
#        f_score_list.append(f_score)
    
        
def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

def sgd_updates_mom(cost,params,mom_param,epoch,squared_filter_length_limit,learning_rate):
    mom_epoch_interval=mom_param['interval']
    mom_end=mom_param['end']
    mom_start=mom_param['start']
    gparams = []
    for param in params:
        # Use the right cost function here to train with or without dropout.
        gparam = T.grad(cost, param)
        gparams.append(gparam)

    # ... and allocate mmeory for momentum'd versions of the gradient
    gparams_mom = []
    for param in params:
        gparam_mom = theano.shared(numpy.zeros(param.get_value(borrow=True).shape,
            dtype=theano.config.floatX))
        gparams_mom.append(gparam_mom)

    # Compute momentum for the current epoch
    mom = ifelse(epoch < mom_epoch_interval,
            mom_start*(1.0 - epoch/mom_epoch_interval) + mom_end*(epoch/mom_epoch_interval),mom_end)
    mom = ifelse(epoch < mom_epoch_interval,
            mom_start*(1.0 - epoch/mom_epoch_interval) + mom_end*(epoch/mom_epoch_interval),
            mom_end)


    # Update the step direction using momentum
    updates = OrderedDict()
    for gparam_mom, gparam in zip(gparams_mom, gparams):
        updates[gparam_mom] = mom * gparam_mom - (1. - mom) * learning_rate * gparam

    # ... and take a step along that direction
    for param, gparam_mom in zip(params, gparams_mom):
        stepped_param = param + updates[gparam_mom]
        if param.get_value(borrow=True).ndim == 2:
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(squared_filter_length_limit))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param
    return updates

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
            tmp=T.cast(tmp,theano.config.floatX)
            #print param.type,tmp.type
            updates[param] = tmp
        else:
            updates[param] = stepped_param
            #print param.type,stepped_param.type
    return updates 

if __name__=='__main__':
    '''
    from optparse import OptionParser as op
    usage="usage: %prog [options]" 
    parser=op(usage=usage)
    
    parser.add_option('--data','--processed_data',type='string',\
                        dest='data',help='处理之后的数据')
    parser.add_option('--flr','--pretrain-lr',type='float',\
                        dest='finetune_lr',help='预训练学习率')
    parser.add_option('--batchsize','--batchsize',type='int',\
                        dest='batchsize',help='batchsize')
    parser.add_option('--nn-structure','--nn-structure',type='string',\
                        dest='nn_structure',help='深度学习模型结构')
    parser.add_option('--corrupt-level','--corrupt-level',type='string',\
                        dest='corrupt_level',help='对于每一层数据的破坏比例')
    (options, args) = parser.parse_args()    
    '''
    sda_train()