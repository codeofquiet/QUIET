# -*- coding: utf-8 -*-
from .data_reader import CRDataReader,MRDataReader,SUBJDataReader,MPQADataReader,SSTDataReader,TRECDataReader
from .data import get_lookup_table
# from loadmydata import *
import os
import codecs
import numpy as np
from keras.utils import to_categorical

##根据配置获取不同的数据集的reader
def setup(opt):
    dir_path = os.path.join(opt.datasets_dir, opt.dataset_name)
    if(opt.dataset_name == 'CR'):
        reader = CRDataReader(dir_path)
    if(opt.dataset_name == 'MR'):
        reader = MRDataReader(dir_path)
    if(opt.dataset_name == 'SUBJ'):
        reader = SUBJDataReader(dir_path)
    if(opt.dataset_name == 'MPQA'):
        reader = MPQADataReader(dir_path)
    if(opt.dataset_name == 'SST_2'):
        ###########分类选择的是这个
        dir_path = os.path.join(opt.datasets_dir, 'SST')
        reader = SSTDataReader(dir_path, nclasses = 2)
        ###########
    if(opt.dataset_name == 'SST_5'):
        dir_path = os.path.join(opt.datasets_dir, 'SST')
        reader = SSTDataReader(dir_path, nclasses = 5)
    if(opt.dataset_name == 'TREC'):
        reader = TRECDataReader(dir_path)
       
   
    return reader

def get_sentiment_dic_training_data(reader, opt):
    word2id = reader.embedding_params['word2id']
    file_name = opt.sentiment_dic_file
    pretrain_x = []
    pretrain_y = []
    with codecs.open(file_name, 'r') as f:
        for line in f:
            word, polarity = line.split()
            if word in word2id:
                word_id = word2id[word]
                pretrain_x.append([word_id]* reader.max_sentence_length)
                pretrain_y.append(int(float(polarity)))
    
    pretrain_x = np.asarray(pretrain_x)
    pretrain_y = to_categorical(pretrain_y)
    return pretrain_x, pretrain_y

def process_embedding(reader,opt):
    opt.max_sequence_length = reader.get_max_sentence_length()####qnn/dataset/classification/data_reader.py class DataReader def get_max_sentence_length
    if  opt.wordvec_path == 'random':
        opt.random_init = True
    else:
        opt.random_init = False
        ###正交归一化
        orthonormalized = (opt.wordvec_initialization == "orthonormalized")
        
        #0104# embedding_params = reader.get_word_embedding(opt.wordvec_path,orthonormalized=orthonormalized)
        ###把params = {'word2id':word2id, 'word_vec':word_vec, 'wvec_dim':wvec_dim,'word_complex_phase':word_complex_phase,'id2word':id2word,"id2idf":idfs}存入到params中
        ###生成对应词的词向量 是50维的
        ###qnn/dataset/classification/data_reader.py class-DataReader def-et_word_embedding
        
        
        ############改为我的lookup-table
        #0104# opt.lookup_table = get_lookup_table(embedding_params)
        opt.lookup_table =[]
        #opt.lookup_table = get_my_lookup_table()
        ############
    
        ####可能就是一个table？
        ###qnn/dataset/classification/data.py def get_lookup_table
        #0104#opt.idfs = embedding_params["id2idf"]

        
    # print(embedding_params['word2id'])
        
    opt.nb_classes = reader.nb_classes
    return opt

    
    