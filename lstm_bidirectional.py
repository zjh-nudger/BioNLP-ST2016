# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 09:38:20 2015

@author: tanfan.zjh
"""
from collections import OrderedDict
import cPickle as pkl
import sys
import time

import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import mlee
import constant

datasets = {constant.dataset: (mlee.load_data, mlee.prepare_data)}

# Set the random number generators' seeds for consistency
SEED = 123
numpy.random.seed(SEED)

##格式转换floatX
def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)


#打乱顺序
def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    n:number of samples
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)

#prepare_data,load_data
def get_dataset(name):
    return datasets[name][0], datasets[name][1]


def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params


def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=0.5, n=1,
                                        dtype=state_before.dtype)),
                         state_before * 0.5)
    return proj

#format the parametes(tanfan.zjh)
def _p(pp, name):
    return '%s_%s' % (pp, name)


#4324 X 200

def init_params(options):
    """
    Global (not LSTM) parameter. For the embeding and the classifier.
    """
    params = OrderedDict()#按键排序
    # embedding：n_words X dim_proj
    '''
    randn = numpy.random.rand(numpy.shape(embedding_trained)[0],
                              options['dim_proj'])#dim_proj:word embedding dimention or lstm hidden units
    '''
    #word_embedding:随机初始化。替换为已训练好的word_embedding
    #params['Wemb'] = (0.01 * randn).astype(config.floatX)
    embedding_trained = pkl.load(open('data/'+options['dataset']+'_emb.pkl'))
    print numpy.shape(embedding_trained)
    params['Wemb'] = embedding_trained.astype(config.floatX)
    params = get_layer(options['encoder'])[0](options,
                                              params,
                                              prefix=options['encoder'])#encoder:lstm
    # classifier
    params['U'] = 0.01 * numpy.random.randn(options['dim_proj'] * 2,
                                            options['ydim']).astype(config.floatX)
    params['b'] = numpy.zeros((options['ydim'],)).astype(config.floatX)
    #print params
    return params


def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params


def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def get_layer(name):
    fns = layers[name]
    return fns


def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(config.floatX)


def param_init_lstm(options, params, prefix='lstm'):
    """
    Init the LSTM parameter:

    :see: init_params
    """
    W = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    #W:dim_proj X 4*dim_proj
    params[_p(prefix, 'W')] = W #lstm_W:W;params:init_param(OrderedDict)
    U = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    #W:dim_proj X 4*dim_proj    
    params[_p(prefix, 'U')] = U
    b = numpy.zeros((4 * options['dim_proj'],))
    params[_p(prefix, 'b')] = b.astype(config.floatX)
    '''
    # for bidirectional
    W_1 = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    #W:dim_proj X 4*dim_proj
    params[_p(prefix, 'W_1')] = W_1 #lstm_W:W;params:init_param(OrderedDict)
    U_1 = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    #W:dim_proj X 4*dim_proj    
    params[_p(prefix, 'U_1')] = U_1
    b_1 = numpy.zeros((4 * options['dim_proj'],))
    params[_p(prefix, 'b_1')] = b_1.astype(config.floatX)
    '''
    return params

    

'''
state_below : embedding as initializatio
'''
def lstm_layer(tparams, state_below,state_below_1, options, prefix='lstm', mask=None):
    nsteps = state_below.shape[0]#句子长度（单词个数）
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]#batch_size
    else:
        n_samples = 1 

    assert mask is not None

    # for the parameters
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(m_, x_, h_, c_):
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_

        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_proj'])) #input gate
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_proj'])) #forget gate
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_proj'])) #output gate
        c = tensor.tanh(_slice(preact, 3, options['dim_proj'])) #cell state

        c = f * c_ + i * c #update for new cell state
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c) #hidden value
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    def _step_1(m_, x_, h_, c_):
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_

        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_proj'])) #input gate
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_proj'])) #forget gate
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_proj'])) #output gate
        c = tensor.tanh(_slice(preact, 3, options['dim_proj'])) #cell state

        c = f * c_ + i * c #update for new cell state
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c) #hidden value
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c
        
    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])
    state_below_1 = (tensor.dot(state_below_1, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])

    dim_proj = options['dim_proj']

    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj),## h
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj)],## c
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    # for bidirectional lstm
    rval_reverse, updates = theano.scan(_step_1,
                                sequences=[mask, state_below_1],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj),## h
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj)],## c
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    #return rval[0] # h
    return theano.tensor.concatenate([rval[0],rval_reverse[0]],axis=2)


# ff: Feed Forward (normal neural net), only useful to put after lstm
#     before the classifier.
layers = {'lstm': (param_init_lstm, lstm_layer)}


def sgd(lr, tparams, grads, x, mask, y, cost):
    """ Stochastic Gradient Descent

    :note: A more complicated version of sgd then needed.  This is
        done like that for adadelta and rmsprop.

    """
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function([x, mask, y], cost, updates=gsup,
                                    name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update


def adadelta(lr, tparams, grads, x, x_1,mask, y, cost):
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
        
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, x_1,mask, y], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, x, mask, y, cost):
    """
    A variant of  SGD that scales the step size by running average of the
    recent step norms.

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [Hint2014]_.

    .. [Hint2014] Geoff Hinton, *Neural Networks for Machine Learning*,
       lecture 6a,
       http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * numpy_floatX(0.),
                           name='%s_updir' % k)
             for k, p in tparams.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(tparams.values(), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update


def build_model(tparams, options):
    trng = RandomStreams(SEED)

    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))

    x = tensor.matrix('x', dtype='int64')#单词索引，固定句子长度?
    x_1 = tensor.matrix('x_1',dtype = 'int64')
    mask = tensor.matrix('mask', dtype=config.floatX)
    y = tensor.vector('y', dtype='int64')

    n_timesteps = x.shape[0]#句子长度
    n_samples = x.shape[1]

    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps,
                                                n_samples,
                                                options['dim_proj']])
    emb_1 = tparams['Wemb'][x_1.flatten()].reshape([n_timesteps,
                                                n_samples,
                                                options['dim_proj']])
    proj = get_layer(options['encoder'])[1](tparams, emb,emb_1, options,
                                            prefix=options['encoder'],
                                            mask=mask)
    if options['encoder'] == 'lstm':
        proj = (proj * mask[:, :, None]).sum(axis=0)
        #print 'type:::::::'
        #print type(proj)
        #proj = (proj * mask[:,:,None]).max(axis=0)
        proj = proj / mask.sum(axis=0)[:, None] #mean pooling
        #proj = (proj * mask[:,:,None])
    if options['use_dropout']:
        proj = dropout_layer(proj, use_noise, trng)

    pred = tensor.nnet.softmax(tensor.dot(proj, tparams['U']) + tparams['b'])

    f_pred_prob = theano.function([x,x_1, mask], pred, name='f_pred_prob')
    f_pred = theano.function([x, x_1,mask], pred.argmax(axis=1), name='f_pred')

    off = 1e-8
    if pred.dtype == 'float16':
        off = 1e-6

    cost = -tensor.log(pred[tensor.arange(n_samples), y] + off).mean()

    return use_noise, x,x_1, mask, y, f_pred_prob, f_pred, cost


def pred_probs(f_pred_prob, prepare_data, data, iterator, verbose=False):
    """ If you want to use a trained model, this is useful to compute
    the probabilities of new examples.
    """
    n_samples = len(data[0])
    probs = numpy.zeros((n_samples, 2)).astype(config.floatX)

    n_done = 0

    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None)
        x_1, _, _ = prepare_data([data[0][t][::-1] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None)
        pred_probs = f_pred_prob(x,x_1, mask)
        probs[valid_index, :] = pred_probs

        n_done += len(valid_index)
        if verbose:
            print '%d/%d samples classified' % (n_done, n_samples)

    return probs


def pred_error(f_pred, prepare_data, data, iterator, verbose=False):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    data:train, valid or test
    """
    valid_err = 0
    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index])
        x_1,_,_ = prepare_data([data[0][t][::-1] for t in valid_index],
                               numpy.array(data[1])[valid_index])
        preds = f_pred(x,x_1, mask)
        targets = numpy.array(data[1])[valid_index]
        valid_err += (preds == targets).sum()
    valid_err = 1. - numpy_floatX(valid_err) / len(data[0])

    return valid_err

import evaluate
def simple_evaluate(f_pred,prepare_data,test_set):
    x,mask,y = prepare_data(test_set[0],test_set[1])
    x_1 = [x[t][::-1] for t in xrange(len(x))]
    prediction = f_pred(x,x_1,mask)
    precision,recall,f_score,micro_avg = evaluate.simple_evaluate(prediction,test_set)
    return precision,recall,f_score,micro_avg

import constant
def train_lstm(
    dim_proj=200,  # word embeding dimension and LSTM number of hidden units.
    max_epochs=5000,  # The maximum number of epoch to run
    dispFreq=1,  # Display to stdout the training progress every N updates
    decay_c=0.,  # Weight decay for the classifier applied to the U weights.classification weights
    lrate=0.0001,  # Learning rate for sgd (not used for adadelta and rmsprop)
    optimizer=adadelta,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
    encoder='lstm',  # TODO: can be removed must be lstm.
    saveto='data/lstm_model.npz',  # The best model will be saved there
    validFreq=1,  # Compute the validation error after this number of update.
    saveFreq=10,  # Save the parameters after every saveFreq updates
    batch_size=256,  # The batch size during training.
#    valid_batch_size=256,  # The batch size used for validation/test set.
    dataset=constant.dataset,

    # Parameter for extra option
    noise_std=0.,
    use_dropout=False,  # if False slightly faster, but worst test error
                       # This frequently need a bigger model.
    reload_model=None  # Path to a saved model we want to start from.
):

    # Model options locals 定义的局部变量
    model_options = locals().copy()
    print "model options", model_options

    load_data, prepare_data = get_dataset(dataset)

    print 'Loading data'
    train, _ , test = load_data(valid_portion=0)
    ydim = numpy.max(train[1]) + 1

    model_options['ydim'] = ydim#分类类别个数

    print 'Building model'
    # This create the initial parameters as numpy ndarrays.
    # Dict name (string) -> numpy ndarray
    params = init_params(model_options)

    if reload_model:
        load_params('lstm_model.npz', params)

    # This create Theano Shared Variable from the parameters.
    # Dict name (string) -> Theano Tensor Shared Variable
    # params and tparams have different copy of the weights.
    tparams = init_tparams(params)

    # use_noise is for dropout
    (use_noise, x,x_1, mask,
     y, f_pred_prob, f_pred, cost) = build_model(tparams, model_options)

    if decay_c > 0.:
        decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')#softmax层权重U权重衰减
        weight_decay = 0.
        weight_decay += (tparams['U'] ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    f_cost = theano.function([x,x_1, mask, y], cost, name='f_cost')

    grads = tensor.grad(cost, wrt=tparams.values())
    f_grad = theano.function([x,x_1, mask, y], grads, name='f_grad')

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = optimizer(lr, tparams, grads,
                                        x,x_1, mask, y, cost)

    print 'Optimization'
    #固定
#    kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
    kf_test = get_minibatches_idx(len(test[0]), batch_size)

    print "%d train examples" % len(train[0])
#    print "%d valid examples" % len(valid[0])
    print "%d test examples" % len(test[0])

    history_errs = []
    best_p = None

    if validFreq == -1:
        validFreq = len(train[0]) / batch_size
    if saveFreq == -1:
        saveFreq = len(train[0]) / batch_size

    uidx = 0  # the number of update done
    start_time = time.time()
    try:
        for eidx in xrange(max_epochs):#for each iteration
            n_samples = 0

            # Get new shuffled index for the training set.
            #每次迭代都打乱训练数据顺序
            kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)

            for _, train_index in kf:#train_index:random,type:list(of index)
                uidx += 1
                use_noise.set_value(1.)#?????????????

                # Select the random examples for this minibatch
                y = [train[1][t] for t in train_index]
                x = [train[0][t]for t in train_index]
                x_1 = [train[0][t][::-1] for t in train_index]
                # Get the data in numpy.ndarray format
                # This swap the axis!
                # Return something of shape (minibatch maxlen, n samples)
                x, mask, y = prepare_data(x, y)
                x_1,_,_ = prepare_data(x_1,y)#[x[t][::-1] for t in xrange(x)]
                n_samples += x.shape[1]

                cost = f_grad_shared(x, x_1,mask, y)
                f_update(lrate)
            
            if numpy.isnan(cost) or numpy.isinf(cost):
                print 'bad cost detected: ', cost
                return 1., 1., 1.
            
            if numpy.mod(eidx, dispFreq) == 0:#uid:add at each update
                print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost
            
            if saveto and numpy.mod(eidx, saveFreq) == 0:
                print 'Saving...',
                if best_p is not None:
                    params = best_p
                else:
                    params = unzip(tparams)
                numpy.savez(saveto, history_errs=history_errs, **params)
                pkl.dump(model_options, open('%s.pkl' % saveto, 'wb'), -1)
                print 'Done'
            
                #没多少次更新之后进行验证
            if numpy.mod(eidx, validFreq) == 0:
                use_noise.set_value(0.)
                train_err = pred_error(f_pred, prepare_data, train, kf)
#                valid_err = pred_error(f_pred, prepare_data, valid,
#                                       kf_valid)
                test_err = pred_error(f_pred, prepare_data, test, kf_test)
                
                p,r,f,m = simple_evaluate(f_pred,prepare_data,test)     
                print ('Train ', round(train_err,3),'Test ', round(test_err,3),
                       'precison:',round(p,3),'recall:',round(r,3),'f_score:',round(f,3),
                       'micro:',round(m,3))
                #x,mask,y = prepare_data(test[0],test[1])
                #prob = f_pred_prob(x,mask)  
                #print prob
                
                history_errs.append(f)#f_score

                if (best_p is None or
                    f >= numpy.array(history_errs).max()):

                    best_p = unzip(tparams)

               # print ('Train ', train_err, 'Valid ', '',
               #        'Test ', test_err)

    except KeyboardInterrupt:
        print "Training interupted"

    end_time = time.time()
    
    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)

    use_noise.set_value(0.)
    kf_train_sorted = get_minibatches_idx(len(train[0]), batch_size)
    train_err = pred_error(f_pred, prepare_data, train, kf_train_sorted)
#    valid_err = pred_error(f_pred, prepare_data, valid, kf_valid)
    test_err = pred_error(f_pred, prepare_data, test, kf_test)

    print 'Train ', train_err, 'Valid ', '', 'Test ', test_err
    if saveto:
        numpy.savez(saveto, train_err=train_err,
                    valid_err='', test_err=test_err,
                    history_errs=history_errs, **best_p)
    print 'The code run for %d epochs, with %f sec/epochs' % (
        (eidx + 1), (end_time - start_time) / (1. * (eidx + 1)))
    print >> sys.stderr, ('Training took %.1fs' %
                          (end_time - start_time))
    return train_err, '', test_err


if __name__ == '__main__':
    # See function train for all possible parameter and there definition.
    train_lstm(
        max_epochs=1000
    )
