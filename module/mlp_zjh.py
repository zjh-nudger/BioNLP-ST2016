# -*- coding: utf-8 -*-
"""
Created on Sun May 10 11:30:49 2015

@author: zjh
"""
import theano,numpy
import theano.tensor as T

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
    
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, activation, W=None, b=None):

        self.input = input
        self.activation = activation

        if W is None:            
            if activation.func_name == "ReLU":
                W_values = numpy.asarray(0.01 * rng.standard_normal(size=(n_in, n_out)), dtype=theano.config.floatX)
            else:                
                W_values = numpy.asarray(rng.uniform(low=-numpy.sqrt(6. / (n_in + n_out)), high=numpy.sqrt(6. / (n_in + n_out)),
                                                     size=(n_in, n_out)), dtype=theano.config.floatX)
            W = theano.shared(value=W_values, name='W',borrow=True)        
        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b',borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b

        self.output = (lin_output if activation is None else activation(lin_output))
    
        self.params = [self.W, self.b]

def _dropout_from_layer(rng, layer, p):
    srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))
    mask = srng.binomial(n=1, p=1-p, size=layer.shape)
    output = layer * T.cast(mask, theano.config.floatX)
    return output

class DropoutHiddenLayer(HiddenLayer):
    def __init__(self, rng, input, n_in, n_out,
                 activation, dropout_rate, W=None, b=None):
        super(DropoutHiddenLayer, self).__init__(
                rng=rng, input=input, n_in=n_in, n_out=n_out, W=W, b=b,
                activation=activation)

        self.output = _dropout_from_layer(rng, self.output, p=dropout_rate)


class MLPDropout(object):
    """A multilayer perceptron with dropout"""
    def __init__(self,rng,input,layer_sizes,dropout_rates,activations):
        # Set up all the hidden layers
        self.weight_matrix_sizes = zip(layer_sizes, layer_sizes[1:])# [(1,2),(2,3),(3,4)]
        self.layers = []
        self.dropout_layers = []
        self.activations = activations
        next_layer_input = input
        next_dropout_layer_input = _dropout_from_layer(rng, input, p=dropout_rates[0])
        layer_counter = 0
        for n_in, n_out in self.weight_matrix_sizes[:-1]:
            next_dropout_layer = DropoutHiddenLayer(rng=rng,
                    input=next_dropout_layer_input,
                    activation=activations[layer_counter],
                    n_in=n_in, n_out=n_out,dropout_rate=dropout_rates[layer_counter])
            self.dropout_layers.append(next_dropout_layer)
            next_dropout_layer_input = next_dropout_layer.output

            # Reuse the parameters from the dropout layer here, in a different
            # path through the graph.
            next_layer = HiddenLayer(rng=rng,
                    input=next_layer_input,
                    activation=activations[layer_counter],
                    # scale the weight matrix W with (1-p)
                    W=next_dropout_layer.W * (1 - dropout_rates[layer_counter]),
                    b=next_dropout_layer.b,
                    n_in=n_in, n_out=n_out)
            self.layers.append(next_layer)
            next_layer_input = next_layer.output
            layer_counter += 1
        
        # Set up the output layer
        n_in, n_out = self.weight_matrix_sizes[-1]
        dropout_output_layer = LogisticRegression(
                input=next_dropout_layer_input,
                n_in=n_in, n_out=n_out)
        self.dropout_layers.append(dropout_output_layer)

        # Again, reuse paramters in the dropout output.
        output_layer = LogisticRegression(
            input=next_layer_input,
            # scale the weight matrix W with (1-p)
            W=dropout_output_layer.W * (1 - dropout_rates[-1]),
            b=dropout_output_layer.b,
            n_in=n_in, n_out=n_out)
        self.layers.append(output_layer)

        # Use the negative log likelihood of the logistic regression layer as
        # the objective.
        self.dropout_negative_log_likelihood = self.dropout_layers[-1].negative_log_likelihood
        self.dropout_errors = self.dropout_layers[-1].errors
        self.dropout_f_score = self.dropout_layers[-1].f_score        
        
        self.negative_log_likelihood = self.layers[-1].negative_log_likelihood
        self.errors = self.layers[-1].errors
        self.f_score = self.layers[-1].f_score  
        
        # Grab all the parameters together.
        self.params = [ param for layer in self.dropout_layers for param in layer.params ]
        
    def predict(self, new_data):
        next_layer_input = new_data
        for i,layer in enumerate(self.layers):
            if i<len(self.layers)-1:
                next_layer_input = self.activations[i](T.dot(next_layer_input,layer.W) + layer.b)
            else:
                p_y_given_x = T.nnet.softmax(T.dot(next_layer_input, layer.W) + layer.b)
        y_pred = T.argmax(p_y_given_x, axis=1)
        return y_pred
    #predict probablity for every class
    def predict_p(self, new_data):
        next_layer_input = new_data
        for i,layer in enumerate(self.layers):
            if i<len(self.layers)-1:
                next_layer_input = self.activations[i](T.dot(next_layer_input,layer.W) + layer.b)
            else:
                p_y_given_x = T.nnet.softmax(T.dot(next_layer_input, layer.W) + layer.b)
        return p_y_given_x

class LogisticRegression(object):

    def __init__(self, input, n_in, n_out, W=None, b=None):
    
        if W is None:
            self.W = theano.shared(
                    value=numpy.zeros((n_in, n_out), dtype=theano.config.floatX),
                    name='W')
        else:
            self.W = W

        if b is None:
            self.b = theano.shared(
                    value=numpy.zeros((n_out,), dtype=theano.config.floatX),
                    name='b')
        else:
            self.b = b

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        #m = [1000. for i in xrange(20)]
        #m.insert(0,1.)
        #p_y_gigen_x_1 = T.mul(self.p_y_given_x,m)

        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
    
    def f_score(self,y,label):
        #print dir(x)
        y=T.cast(y,'int32')
        new_y_pred=T.sub(self.y_pred,label)
        new_y=T.sub(y,label)
        
        pre_pos_num=new_y_pred.shape[0]-new_y_pred.nonzero()[0].shape[0]#预测的正例个数
        
        real_pos=new_y.shape[0]-new_y.nonzero()[0].shape[0]
        
        new_y_pred=T.set_subtensor(new_y_pred[new_y_pred.nonzero()[0]],1)
        new_y=T.set_subtensor(new_y[new_y.nonzero()[0]],2)

        r=T.neq(new_y_pred,new_y)
        true_pos=self.y_pred.shape[0]-r.sum()
        #printed_recall=theano.printing.Print('rec:')(pre_pos_num)
        #printed=theano.printing.Print('pre:')(real_pos)
        precision=true_pos / (T.cast(pre_pos_num,'float32')+0.0000001)
        recall=true_pos / (T.cast(real_pos,'float32')+0.0000001)
        
        f_score=(2 * precision * recall) / (precision + recall)
        
        #return f_score,precision,recall
        return f_score,precision,recall
        
        