#coding=gbk

import os,time,numpy,theano,sys
sys.path.append(os.path.abspath('..'))
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from utils.data_manager import load_bionlp_data


class dA(object):

    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        input=None,
        n_visible=784,
        n_hidden=500,
        W=None,
        bhid=None,
        bvis=None,
        activation=T.tanh
    ):

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.activation=activation
        # create a Theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # note : W' was written as `W_prime` and b' as `b_prime`
        if not W:
            # W is initialized with `initial_W` which is uniformely sampled
            # from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-1 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high=1 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            if activation==theano.tensor.nnet.sigmoid:
                initial_W *=4

            W = theano.shared(value=initial_W, name='W',
                              borrow=True)

        if not bvis:
            bvis = theano.shared(
                value=numpy.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                name='b_prime',
                borrow=True
            )

        if not bhid:
            bhid = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )

        self.W = W
        self.b = bhid
        self.b_prime = bvis
        self.W_prime = self.W.T

        self.theano_rng = theano_rng
        # if no input is given, generate a variable representing the input
        if input is None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.dmatrix(name='input')
        else:
            self.x = input
        #print dir(shape(self.W))
        #print shape(self.W).eval(),shape(self.b).eval(),shape(self.b_prime).eval()
        #print self.b.broadcastable,self.b_prime.broadcastable
        self.params = [self.W, self.b, self.b_prime]
    # end-snippet-1


    #对输入进行一定的干扰
    def get_corrupted_input(self, input, corruption_level):
        #param p：产生1的概率
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                       dtype=theano.config.floatX) * input

    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer """
        result=self.activation(T.dot(input, self.W) + self.b)

        return result

    def get_reconstructed_input(self, hidden):
        """Computes the reconstructed input given the values of the
        hidden layer

        """
        #print self.activation
        result=self.activation(T.dot(hidden, self.W_prime) + self.b_prime)

        return result

    def get_cost_updates(self, corruption_level, learning_rate,cost_function_name):
        """
            This function computes the cost and the updates for one trainng
            step of the dA
        """
        #print str(self.activation)
        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        #print self.activation
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        cost=None
        
        if cost_function_name=='cross_entropy':
            #print 'cross_entropy..'
            L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1-z), axis=1)
            cost = T.mean(L) #所有节点求和，然后所有batch_size求平均
        if cost_function_name=='sqr_error':
            L=(T.sum(T.square(T.abs_(self.x-z))/2.,axis=0))/T.cast(T.shape(self.x)[0],'float32')
            #theano.printing.debugprint(obj=cost,print_type=True)
            #printdebug.debugprint(cost)
            cost=T.mean(L)
            
        T.cast(cost, 'float32')
        #print cost
        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        return (cost, updates)

    def get_restructed_error(self,input):
        hidden_value=self.get_hidden_values(input)
        restructed_output=self.get_reconstructed_input(hidden_value)
        error=numpy.abs(input - restructed_output)
        #print (numpy.shape(error)).eval()
        return T.mean(T.sum(error,axis=1),axis=0)#每个节点的平均误差

def test_dA(learning_rate=0.2,
            training_epochs=10,
            dataset=['../train.txt ',None,'../test.txt'],
            batch_size=1):
    datasets = load_bionlp_data(dataset)
    train_set_x, train_set_y = datasets[0]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    # allocate symbolic variables for the data
    index = T.lscalar('index')    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images

    #####################################
    # BUILDING THE MODEL CORRUPTION 30% #
    #####################################

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = dA(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=x,
        n_hidden=4,
        n_visible=4
    )

    cost, updates = da.get_cost_updates(
        corruption_level=0.0,
        learning_rate=learning_rate
    )

    train_da = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )

    start_time = time.clock()

    ############
    # TRAINING #
    ############

    # go through training epochs
    for epoch in xrange(training_epochs):
        # go through trainng set
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(train_da(batch_index))

        #print 'Training epoch %d, %fcost \n' % (epoch, numpy.mean(c))
        print da.get_hidden_values(train_set_x).eval()
        print da.get_restructed_error(train_set_x).eval()
    end_time = time.clock()
    print '耗时:%f'%(end_time-start_time)


if __name__ == '__main__':
    test_dA()
