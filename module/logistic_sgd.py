#!/usr/bin/env python
#coding=gbk

__docformat__ = 'restructedtext en'

import os,sys,time,numpy,theano
import theano.tensor as T
sys.path.append(os.path.abspath('..'))
from utils import data_manager as dm


class LogisticRegression(object):
    """
        ��softmmax�ع������������
    """

    def __init__(self, input, n_in, n_out):
        # ��Ȩ�ؾ���n_in X n_out����ʼ��Ϊ0
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX #GPU config
            ),
            name='W',
            borrow=True
        ) # ������� borrow name

        # ��ƫ��b��n_outά��һ����������Ϊ0
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        #Modelģ��
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b) #����

        #���ظ������ֵ��Ӧ������������ʵ�����ൽ����������һ����
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        #��ģ�͵Ĳ���
        self.params = [self.W, self.b]

    #��������Ȼ��������ʧ��������
    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
    '''
    def predict(self,input):
        p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        y_pred = T.argmax(p_y_given_x, axis=1)
        return p_y_given_x,y_pred
    '''
    def errors(self, y):
        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        #����ǩ������
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()



def sgd_optimization(learning_rate=0.13, 
                     n_epochs=1000,
                     dataset=['../train.txt',None,'../test.txt'],
                     batch_size=1):
    datasets = dm.load_bionlp_data(dataset)

    train_set_x, train_set_y = datasets[0]
    if datasets[1]!=None:
        valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # ����MiniBatch�ĸ�����ÿ��minibatch������һ�β���
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    # ͨ��������ȡ����
    index = T.lscalar()  # index to a [mini]batch

    # x,y : minibatch
    x = T.matrix('x')  # data
    y = T.ivector('y')  # labels,һά����

    # construct the logistic regression class
    classifier = LogisticRegression(input=x, n_in=28 * 28, n_out=10)

    # ��������Ȼ��������ʧ����
    cost = classifier.negative_log_likelihood(y)

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    # ����ģ������֤������Լ������..
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # �ݶȼ���
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    # ���²���
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
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

    ###############
    # TRAIN MODEL #
    ###############
    print '... training the model'
    patience = 5000  
    patience_increase = 2 
    improvement_threshold = 0.995 
    validation_frequency = min(n_train_batches, patience / 2)

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        # ÿһ�ε���...
        epoch = epoch + 1
        #����ÿ�ε���������һ��ѵ����
        for minibatch_index in xrange(n_train_batches):
            # ��һ��MiniBatch�е�ƽ����ʧ
            minibatch_avg_cost = train_model(minibatch_index)
            #
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # ����֤���ϼ���ƽ����ʧ
                validation_losses = [validate_model(i)
                                     for i in xrange(n_valid_batches)]
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

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [test_model(i)
                                   for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(
                        (
                            '     epoch %i, minibatch %i/%i, test error of'
                            ' best model %f %%'
                        ) %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )
    print 'The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.1fs' % ((end_time - start_time)))

if __name__ == '__main__':
    sgd_optimization()
    #dataset=load_data()
