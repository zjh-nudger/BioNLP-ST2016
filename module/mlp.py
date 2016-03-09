# -*- coding: utf-8 -*-
"""
.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5

"""
__docformat__ = 'restructedtext en'


import os
import sys
import time

import numpy

import theano
import theano.tensor as T

from logistic_sgd import LogisticRegression

sys.path.append(os.path.abspath('..'))
from utils.data_manager import load_bionlp_data

# 隐藏层
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.nnet.sigmoid):
        """

        NOTE : 激活函数用的是Tanh

        :type rng: numpy.random.RandomState
        :param rng: 用于初始化权重矩阵W的随机数生成器

        :type input: theano.tensor.dmatrix
        :param input: 数据:n_example*n_in

        :type n_in: int
        :param n_in: 特征维度

        :type n_out: int
        :param n_out: 隐藏单元节点个数

        :type activation: theano.Op or function
        :param activation:激活函数
        """
        self.input = input
        self.activation=activation

        # 对tanh激活函数，初始化W 从sqrt(-6./(n_in+n_hidden))到 sqrt(6./(n_in+n_hidden))
        # 初始化权重矩阵参数根据激活函数的不同而不同
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX # run on GPU
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b #隐藏层的输入
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        ) #隐藏层的输出
        # 模型的参数，这是输入与隐藏层之间的w与b
        self.params = [self.W, self.b]

    #根据隐藏层的输入，计算出隐藏层的输出
    def get_output(self,input):
        lin_output=T.dot(input,self.W)+self.b
        output = self.activation(lin_output)
        return output

# start-snippet-2
class MLP(object):
    """
    中间层是采用激活函数，而最后一层是一个softmax层
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out):
        """
        :type rng: numpy.random.RandomState
        :param rng: 随机数生成器，初始化权重矩阵w(2),b(2)

        :type input: theano.tensor.TensorType
        :param input: one minibatch

        :type n_in: int
        :param n_in: 输入数据的维度

        :type n_hidden: int
        :param n_hidden: 隐藏层节点个数

        :type n_out: int
        :param n_out: 类别个数

        """

        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
        )

        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )

        # L1正则化项，w(1),w(2)
        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )
        # L2正则化项
        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )
        # MLP的负对数似然函数
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        self.errors = self.logRegressionLayer.errors

        # 整个模型的参数W(1)+W(2)
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params
        # end-snippet-3


def test_mlp(learning_rate=0.05, L1_reg=0.00, L2_reg=0.001, n_epochs=1000,
             dataset='data/dataset_BioNlp2009_GE_95_all.pkl.gz', batch_size=32, n_hidden=2000):
    """
    MLP参数

    :type learning_rate: float
    :param learning_rate: 学习率

    :type L1_reg: float
    :param L1_reg: L1正则化项权重

    :type L2_reg: float
    :param L2_reg: L2正则化项权重

    :type n_epochs: int
    :param n_epochs: 迭代最大次数

    :type dataset: string
    :param dataset: 数据集文件地址


   """
    datasets = load_bionlp_data(dataset)

    # 共享变量
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # 计算MiniBatch的个数
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  #
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    classifier = MLP(
        rng=rng,
        input=x,
        n_in=2577,
        n_hidden=n_hidden,
        n_out=10
    )
    # 损失函数
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    # 在测试集上测试正确率
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )
    # 在验证集上测试正确率
    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # start-snippet-5
    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    # 计算梯度 w(1)\w(2)
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two list the zip A = [a1, a2, a3, a4] and B = [b1, b2, b3, b4] of
    # same length, zip generates a list C of same size, where each element
    # is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-5

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )
                '''
                logger.info('epoch %i, minibatch %i/%i, validation error %f %% \r\n' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    ))
                '''
                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i
                                   in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))
                    '''
                    logger.info(('    epoch %i, minibatch %i/%i, test error of '
                           'best model %f %% \r\n') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))
                    '''

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    '''
    logger.info(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%   ') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))

    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    logger.info('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm'% ((end_time - start_time) / 60.))

    '''
if __name__ == '__main__':
    test_mlp()
