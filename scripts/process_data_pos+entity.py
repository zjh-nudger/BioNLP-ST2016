#coding:gbk
"""
Created on Sun March 9 10:20:31 2016

处理成深度学习模型输入

词性+实体类型特征

@author: zjh
"""
import numpy as np
from optparse import OptionParser as op


def ugly_normalize(vecs):
   normalizers = np.sqrt((vecs * vecs).sum(axis=1))
   normalizers[normalizers==0]=1
   return (vecs.T / normalizers).T

class Embeddings:
   def __init__(self, vecsfile, vocabfile=None, normalize=True):
      if vocabfile is None: vocabfile = vecsfile.replace("npy","vocab")
      self._vecs = np.load(vecsfile)
      self._vocab = file(vocabfile).read().split()
      if normalize:
         self._vecs = ugly_normalize(self._vecs)
      self._w2v = {w:i for i,w in enumerate(self._vocab)}

   @classmethod
   def load(cls, vecsfile, vocabfile=None):
      return Embeddings(vecsfile, vocabfile)

   def word2vec(self, w):
      return self._vecs[self._w2v[w]]


if __name__=='__main__':
    usage="usage: %prog [options]" 
    parser=op(usage=usage)
    parser.add_option('--vec','--vec_file',type='string',\
                        dest='vec_file',help='词向量文件_npy')
    parser.add_option('--vocab','--vocab_file',type='string',\
                        dest='vocab_file',help='词典文件_vocab') 
    parser.add_option('--train','--train_file',type='string',\
                        dest='train_file',help='训练文件')    
    parser.add_option('--test','--test_file',type='string',\
                        dest='test_file',help='测试文件')
    parser.add_option('--output','--output_file',type='string',\
                        dest='output_file',help='测试文件')
#    parser.add_option('--len','--sen_len',type='int',default=3,\
#                        dest='sen_len',help='句子长度')  
    
    
    (options, args) = parser.parse_args()
    
    embedding=Embeddings.load(vecsfile=options.vec_file,
                              vocabfile=options.vocab_file)
    
    word_size=np.shape(embedding._vecs)[1]#向量维度
    
    #zero_vec=np.zeros(word_size)
    
    train_file=open(options.train_file)
    test_file=open(options.test_file)
    
    vocab={}#vocabulary
    index=1
    sen_len_max = 0
    # build the vocab    
    for item in train_file:
        items=item.split()
        if len(items[1:]) > sen_len_max:
            sen_len_max = len(items[1:])
        for item1 in items[1:]:
            if vocab.has_key(item1.lower()):
                continue
            vocab[item1.lower()] = index
            index+=1
    train_file.close()
    for item in test_file:
        items=item.split()
        for item1 in items[2:]:
            if vocab.has_key(item1.lower()):
                continue
            vocab[item1.lower()]=index
            index+=1
    test_file.close()

    print 'There are %d words in train and test dataset'%len(vocab)    
    
    words=np.zeros((len(vocab)+1,word_size),dtype='float32')
    #words[0]=embedding.word2vec('/s')
    count_not_in_vocab = 0
    for item in vocab.iteritems():
        if item[0] in embedding._vocab:
            #print item[1]
            words[item[1]] = embedding.word2vec(item[0])
        else:
            #if there is not the words in vector file
            #random sample the vector
            count_not_in_vocab += 1
            rand_vec=np.random.uniform(-0.25,0.25,word_size)
            words[item[1]] = rand_vec
    print 'There are %d words not in vocab.'%count_not_in_vocab
    
    train_file=open(options.train_file)
            
    train_file=open(options.train_file)    
    import cPickle
    train_list=[]
    for item in train_file:
        items=item.split()
        new_item=[]
        new_item.append(int(items[0]))
        if len(items[1:-2]) <= sen_len_max:        
            for i in items[1:-2]:
                new_item.append(int(vocab[i.lower()]))
        else:
            for i in items[1:sen_len_max+1]:
                new_item.append(int(vocab[i.lower()]))
        if len(new_item[1:]) != sen_len_max:
            for i in xrange(sen_len_max-len(new_item[1:])):
                new_item.append(0)
        new_item.append(int(vocab[items[-2].lower()]))
        new_item.append(int(vocab[items[-1].lower()]))
        train_list.append(new_item)
    train_matrix=np.asarray(train_list,dtype='float32')

    test_file=open(options.test_file)
    test_list=[]
    for item in test_file:
        items=item.split()
        new_item=[]    
        new_item.append(int(items[1]))
        if len(items[2:-2]) <= sen_len_max:        
            for i in items[2:-2]:
                new_item.append(int(vocab[i.lower()]))
        else:
            for i in items[2:sen_len_max+2]:
                new_item.append(int(vocab[i.lower()]))
        if len(new_item[1:]) != sen_len_max:
            for i in xrange(sen_len_max-len(new_item[1:])):
                new_item.append(0)
        new_item.append(int(vocab[items[-2].lower()]))
        new_item.append(int(vocab[items[-1].lower()]))
        test_list.append(new_item)
    test_matrix=np.asarray(test_list,dtype='float32')
    print np.shape(test_matrix)
    print np.shape(train_matrix)
    
    pos_embedding = np.random.uniform(-0.25,0.25,(34,50))
    pos_embedding[0] = np.zeros(50)
    train_pos_file = open(options.train_file+'_pos')
    pos_vocab = {}
    index = 1
    train_pos_list = []
    for item in train_pos_file:
        items=item.strip().split()
        new_item=[]
        if len(items) <= sen_len_max:        
            for i in items:
                if pos_vocab.has_key(item):
                    new_item.append(pos_vocab[i])
                else:
                    pos_vocab[i] = index
                    index += 1
        else:
            for i in items[:sen_len_max+1]:
                if pos_vocab.has_key(item):
                    new_item.append(pos_vocab[i])
                else:
                    pos_vocab[i] = index
                    index += 1
        if len(new_item) != sen_len_max:
            for i in xrange(sen_len_max-len(new_item)):
                new_item.append(0)
        new_item.append(0)
        new_item.append(0)
        train_pos_list.append(new_item)
    train_pos_matrix=np.asarray(train_pos_list,dtype='float32')
    
    test_pos_file = open(options.test_file+'_pos')
    test_pos_list = []
    for item in test_pos_file:
        items=item.strip().split()
        new_item=[]
        if len(items) <= sen_len_max:        
            for i in items:
                if pos_vocab.has_key(item):
                    new_item.append(pos_vocab[i])
                else:
                    pos_vocab[i] = index
                    index += 1
        else:
            for i in items[:sen_len_max+1]:
                if pos_vocab.has_key(item):
                    new_item.append(pos_vocab[i])
                else:
                    pos_vocab[i] = index
                    index += 1
        if len(new_item) != sen_len_max:
            for i in xrange(sen_len_max-len(new_item)):
                new_item.append(0)
        new_item.append(0)
        new_item.append(0)
        test_pos_list.append(new_item)
    test_pos_matrix=np.asarray(test_pos_list,dtype='float32')

###############################################################
################################################################

    #print vocab
    #print test_matrix
    #print train_matrix
    #/home/liuxiaoming/dl_zjh/data/arguments/pubmed/data.p
    cPickle.dump((train_matrix,train_pos_matrix,
                  test_matrix,test_pos_matrix,
                  words,pos_embedding),
                 open(options.output_file,'wb'))