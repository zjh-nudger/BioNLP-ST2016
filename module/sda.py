#coding:gbk
"""
Created on Mon May 18 14:01:41 2015

@author: zjh
"""
import numpy,theano,sys,os
sys.path.append(os.path.abspath('..'))
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from logistic_sgd import LogisticRegression
from mlp_dropout import DropoutHiddenLayer as dpLayer
from dA import dA

class SDA(object):
    
    def __init__(
        self,
        W_sda=[None,None],#���ѱ���Ĳ����Ļ��Ͷ���
        b_sda=[None,None],
        nn_structure=[600,500,500,10], # ����㡢���ز����ڵ�������������
        dropout_rates=[0.5,0.5],
        L1_reg=0,L2_reg=0,
        activation=T.nnet.sigmoid,
        non_static=False,
        wordvec=None
    ):
        #print activation
        self.activation=activation
        self.sigmoid_layers = [] #���ز��б�...
        self.dA_layers = [] #da���б�...
        self.params = [] #������dA��....sigmoid layers...
        self.n_layers = len(nn_structure)-2 #�м������ز�
        
        #random num
        numpy_rng = numpy.random.RandomState(89677)
        theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        self.x = T.matrix('x')#data
        self.y = T.ivector('y')#label
        #print wordvec
        self.Words=theano.shared(value=wordvec,name='Words')
        layer0_input= self.Words[T.cast(self.x.flatten(),dtype="int32")].\
                      reshape((self.x.shape[0],self.x.shape[1]*self.Words.shape[1]))
        #n=layer0_input.shape[1] / numpy.shape(wordvec)[0]
        '''
        #���������...
        avg_vec = numpy.zeros(numpy.shape(wordvec)[0],dtype=theano.config.floatX)
        for i in xrange(5):
            sub_vec=layer0_input[:,i*200:(i+1)*200]
            avg_vec=T.add(sub_vec,avg_vec)
        #avg_vec=avg_vec/5.
        layer0_input=avg_vec
        '''
        # DA���MLP�㹲��Ȩ��..
        for i in xrange(self.n_layers):
            if i == 0:
                input_size = nn_structure[0] #input_size:ÿ�����ز������ڵ����
            else:
                input_size = nn_structure[i]

            if i == 0:
                layer_input = layer0_input#self.x #��������
            else:
                layer_input = self.sigmoid_layers[i-1].output # ��һ���������ǰһ������
                
            #����W\b�����ʼ��..dropout layer
            sigmoid_layer = dpLayer(rng=numpy_rng,
                                    input=layer_input,
                                    n_in=input_size,
                                    W=W_sda[i],
                                    b=b_sda[i],
                                    dropout_rate=dropout_rates[i],
                                    n_out=nn_structure[i+1],
                                    activation=self.activation)
            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer) #���ز��б�

            self.params.extend(sigmoid_layer.params) #mlp����w,b

            # ��hidden layer����Ȩ��
            dA_layer = dA(numpy_rng=numpy_rng,
                          theano_rng=theano_rng,
                          input=layer_input,
                          n_visible=input_size,
                          n_hidden=nn_structure[i+1],
                          W=sigmoid_layer.W, # share weights with sigmoid layer
                                             # W�ǹ��������dAԤѵ��
                          bhid=sigmoid_layer.b,
                          activation=self.activation)
            self.dA_layers.append(dA_layer)

        self.logLayer = LogisticRegression(
            input=self.sigmoid_layers[-1].output,
            n_in=nn_structure[-2],
            n_out=nn_structure[-1]
        )

        self.params.extend(self.logLayer.params)#log���� w,b
    

        L1=0.
        L2=0.
        for layer in self.sigmoid_layers:
            L1+=abs(layer.W).sum()
            L2+=(layer.W ** 2).sum()
            
        self.L1=(L1+abs(self.logLayer.W).sum())
        self.L2_sqr=L2+(self.logLayer.W ** 2).sum()
        
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y) \
                            + L1_reg * self.L1 + L2_reg * self.L2_sqr 
                            
        if non_static:
            self.params.extend([self.Words])

        self.errors = self.logLayer.errors(self.y)