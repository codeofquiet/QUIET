# -*- coding: utf-8 -*-
# from transformers import BertTokenizer
# from transformers import TFBertModel
from enum import auto
import tensorflow as tf
from params import Params
from models import representation as models
from dataset import classification as dataset
from tools import units
from tools.save import save_experiment
# from loadmydata import *
import itertools
import argparse
import keras.backend as K
import numpy as np 
from keras.utils import plot_model, multi_gpu_model
from datagenerator import TrainDataGenerator, getValidData, getXY
import time
from keras.callbacks import Callback, EarlyStopping, TensorBoard

from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score,balanced_accuracy_score
from sklearn.metrics import confusion_matrix
import logging
# import time
import keras

gpu_count = len(units.get_available_gpus())
dir_path, global_logger = units.getLogger()

my_seed = 666  # 939
np.random.seed(my_seed)
import random
random.seed(my_seed)
tf.random.set_random_seed(my_seed)

now = int(round(time.time()*1000))
now = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(now/1000))
file_sar = open("../checkdata/val_prediction_label/normal_sarlabel.txt", mode='a')
file_sar.write('\n'+str(now)+'\n')
file_sent = open("../checkdata/val_prediction_label/normal_sentlabel.txt", mode='a')
file_sent.write('\n'+str(now)+'\n')
file_emo = open("../checkdata/val_prediction_label/normal_emolabel.txt", mode='a')
file_emo.write('\n'+str(now)+'\n')
file_name = open("../checkdata/val_prediction_label/normal_uttNameList.txt", mode='a')
file_name.write('\n'+str(now)+'\n')
file_emo.close()
file_sar.close()
file_sent.close()
file_name.close()

def EvalModel(model):
    result = []
    textFeaList, imageFeaList, audioFeaList,\
        textPhaseList, imagePhaseList, audioPhaseList,\
        sarList, sentList, emoList, weight, namelist = getValidData(datatype='test')
    # TrainDataGenerator(dataset='train', batch_size=params.batch_size, shuffle=True)
    # sent_val_predict, val_predict = model.predict([
    sentoutput, saroutput, emooutput = model.predict([
        textFeaList,
        imageFeaList,
        audioFeaList,
        textPhaseList,
        imagePhaseList,
        audioPhaseList,
        weight])
    # sar_val_predict = 1-np.argmax(saroutput, -1)
    sar_val_predict = np.argmax(saroutput, -1)
    sent_val_predict = np.argmax(sentoutput, -1)
    emo_val_predict = np.argmax(emooutput, -1)
    sar_val_targ = sarList
    sent_val_targ = sentList
    emo_val_targ = emoList
    if len(sar_val_targ.shape) == 2 and sar_val_targ.shape[1] != 1:
        sar_val_targ = np.argmax(sar_val_targ, -1)
    if len(sent_val_targ.shape) == 2 and sent_val_targ.shape[1] != 1:
        sent_val_targ = np.argmax(sent_val_targ, -1)
    if len(emo_val_targ.shape) == 2 and emo_val_targ.shape[1] != 1:
        emo_val_targ = np.argmax(emo_val_targ, -1)

    tn, fp, fn, tp = confusion_matrix(sar_val_targ, sar_val_predict).ravel()
    _specificity = tn / (tn+fp)

    sent_val_bacc = balanced_accuracy_score(sent_val_targ, sent_val_predict)
    sent_val_f1 = f1_score(sent_val_targ, sent_val_predict, average='micro')
    sent_val_recall = recall_score(sent_val_targ, sent_val_predict, average='micro')
    sent_val_precision = precision_score(sent_val_targ, sent_val_predict, average='micro')
    sent_val_acc = accuracy_score(sent_val_targ, sent_val_predict)
    result.append(sent_val_bacc)
    result.append(sent_val_acc)
    result.append(sent_val_f1)
    result.append(sent_val_precision)
    result.append(sent_val_recall)

    sar_val_bacc = balanced_accuracy_score(sar_val_targ, sar_val_predict)
    sar_val_f1 = f1_score(sar_val_targ, sar_val_predict, average='micro')
    sar_val_recall = recall_score(sar_val_targ, sar_val_predict, average='micro')
    sar_val_precision = precision_score(sar_val_targ, sar_val_predict, average='micro')
    sar_val_acc = accuracy_score(sar_val_targ, sar_val_predict)
    result.append(sar_val_bacc)
    result.append(sar_val_acc)
    result.append(sar_val_f1)
    result.append(sar_val_precision)
    result.append(sar_val_recall)
    result.append(_specificity)

    emo_val_bacc = balanced_accuracy_score(emo_val_targ, emo_val_predict)
    emo_val_f1 = f1_score(emo_val_targ, emo_val_predict, average='micro')
    emo_val_recall = recall_score(emo_val_targ, emo_val_predict, average='micro')
    emo_val_precision = precision_score(emo_val_targ, emo_val_predict, average='micro')
    emo_val_acc = accuracy_score(emo_val_targ, emo_val_predict)
    result.append(emo_val_bacc)
    result.append(emo_val_acc)
    result.append(emo_val_f1)
    result.append(emo_val_precision)
    result.append(emo_val_recall)
    print('eval result:')
    print("- sar_val_bacc: %f - sar_vval_acc: %f  — sar_vval_f1: %f — sar_vval_precision: %f — sar_vval_recall: %f — specificity: %f" % (sar_val_bacc, sar_val_acc, sar_val_f1, sar_val_precision, sar_val_recall, _specificity))
    print("- sent_val_bacc: %f - sent_val_acc: %f  — sent_val_f1: %f — sent_val_precision: %f — sent_val_recall: %f" % (sent_val_bacc, sent_val_acc, sent_val_f1, sent_val_precision, sent_val_recall))
    print("- emo_val_bacc: %f - emo_val_acc: %f  — emo_val_f1: %f — emo_val_precision: %f — emo_val_recall: %f" % (emo_val_bacc, emo_val_acc, emo_val_f1, emo_val_precision, emo_val_recall))
    file_name = open("../checkdata/val_prediction_label/normal_uttNameList.txt", mode='a')
    file_name.write(str(namelist)+'\n')

    file_sar = open("../checkdata/val_prediction_label/normal_sarlabel.txt", mode='a')
    file_sar.write('sar_val_predict')
    file_sar.write(str(sar_val_predict)+'\n')
    file_sar.write('sar_val_targ')
    file_sar.write(str(sar_val_targ)+'\n')

    file_sent = open("../checkdata/val_prediction_label/normal_sentlabel.txt", mode='a')
    file_sent.write('sent_val_predict')
    file_sent.write(str(sent_val_predict)+'\n')
    file_sent.write('sent_val_targ')
    file_sent.write(str(sent_val_targ)+'\n')

    file_emo = open("../checkdata/val_prediction_label/normal_emolabel.txt", mode='a')
    file_emo.write('emo_val_predict')
    file_emo.write(str(emo_val_predict)+'\n')
    file_emo.write('emo_val_targ')
    file_emo.write(str(emo_val_targ)+'\n')
    
    file_name.close()
    file_emo.close()
    file_sar.close()
    file_sent.close()

    return result

class Metrics(Callback):
    def __init__(self, valid_data):
        super(Metrics, self).__init__()
        self.textFeaList = valid_data[0]
        self.imageFeaList = valid_data[1]
        self.audioFeaList = valid_data[2]
        self.textPhaseList = valid_data[3]
        self.imagePhaseList = valid_data[4]
        self.audioPhaseList = valid_data[5]
        self.sarList = valid_data[6]
        self.sentList = valid_data[7]
        self.emoList = valid_data[8]
        self.weight = valid_data[9]

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        sent_val_predict, sar_val_predict, emo_val_predict = self.model.predict([
            self.textFeaList,
            self.imageFeaList,
            self.audioFeaList,
            self.textPhaseList,
            self.imagePhaseList,
            self.audioPhaseList,
            self.weight])
        # sar_val_predict = 1-np.argmax(sar_val_predict, -1)
        sar_val_predict = np.argmax(sar_val_predict, -1)
        sent_val_predict = np.argmax(sent_val_predict, -1)
        emo_val_predict = np.argmax(emo_val_predict, -1)
        sar_val_targ = self.sarList
        sent_val_targ = self.sentList
        emo_val_targ = self.emoList
        if len(sar_val_targ.shape) == 2 and sar_val_targ.shape[1] != 1:
            sar_val_targ = np.argmax(sar_val_targ, -1)
        if len(sent_val_targ.shape) == 2 and sent_val_targ.shape[1] != 1:
            sent_val_targ = np.argmax(sent_val_targ, -1)
        if len(emo_val_targ.shape) == 2 and emo_val_targ.shape[1] != 1:
            emo_val_targ = np.argmax(emo_val_targ, -1)

        tn, fp, fn, tp = confusion_matrix(sar_val_targ, sar_val_predict).ravel()
        _specificity = tn / (tn+fp)
        sent_val_bacc = balanced_accuracy_score(sent_val_targ, sent_val_predict)
        sent_val_f1 = f1_score(sent_val_targ, sent_val_predict, average='micro')
        sent_val_recall = recall_score(sent_val_targ, sent_val_predict, average='micro')
        sent_val_precision = precision_score(sent_val_targ, sent_val_predict, average='micro')
        sent_val_acc = accuracy_score(sent_val_targ, sent_val_predict) 
        logs['sent_val_acc'] = sent_val_acc
        logs['sent_val_f1'] = sent_val_f1
        logs['sent_val_recall'] = sent_val_recall
        logs['sent_val_precision'] = sent_val_precision

        sar_val_bacc = balanced_accuracy_score(sar_val_targ, sar_val_predict)
        sar_val_f1 = f1_score(sar_val_targ, sar_val_predict, average='micro')
        sar_val_recall = recall_score(sar_val_targ, sar_val_predict, average='micro')
        sar_val_precision = precision_score(sar_val_targ, sar_val_predict, average='micro')
        sar_val_acc = accuracy_score(sar_val_targ, sar_val_predict) 
        logs['sar_val_acc'] = sar_val_acc
        logs['sar_val_f1'] = sar_val_f1
        logs['sar_val_recall'] = sar_val_recall
        logs['sar_val_precision'] = sar_val_precision
        logs['sar_specificity'] = _specificity

        emo_val_bacc = balanced_accuracy_score(emo_val_targ, emo_val_predict)
        emo_val_f1 = f1_score(emo_val_targ, emo_val_predict, average='micro')
        emo_val_recall = recall_score(emo_val_targ, emo_val_predict, average='micro')
        emo_val_precision = precision_score(emo_val_targ, emo_val_predict, average='micro')
        emo_val_acc = accuracy_score(emo_val_targ, emo_val_predict) 
        logs['emo_val_acc'] = emo_val_acc
        logs['emo_val_f1'] = emo_val_f1
        logs['emo_val_recall'] = emo_val_recall
        logs['emo_val_precision'] = emo_val_precision
        print("- sar_val_bacc: %f - sar_val_acc: %f  — sar_val_f1: %f — sar_val_precision: %f — sar_val_recall: %f — specificity: %f" % (sar_val_bacc, sar_val_acc, sar_val_f1, sar_val_precision, sar_val_recall, _specificity))
        print("- sent_val_bacc: %f - sent_val_acc: %f  — sent_val_f1: %f — sent_val_precision: %f — sent_val_recall: %f" % (sent_val_bacc, sent_val_acc, sent_val_f1, sent_val_precision, sent_val_recall))
        print("- emo_val_bacc: %f - emo_val_acc: %f  — emo_val_f1: %f — emo_val_precision: %f — emo_val_recall: %f" % (emo_val_bacc, emo_val_acc, emo_val_f1, emo_val_precision, emo_val_recall))

        logger.info("- sar_val_bacc: %f - sar_val_acc: %f  — sar_val_f1: %f — sar_val_precision: %f — sar_val_recall: %f — specificity: %f" % (sar_val_bacc, sar_val_acc, sar_val_f1, sar_val_precision, sar_val_recall, _specificity))
        logger.info("- sent_val_bacc: %f - sent_val_acc: %f  — sent_val_f1: %f — sent_val_precision: %f — sent_val_recall: %f" % (sent_val_bacc, sent_val_acc, sent_val_f1, sent_val_precision, sent_val_recall))
        logger.info("- emo_val_bacc: %f - emo_val_acc: %f  — emo_val_f1: %f — emo_val_precision: %f — emo_val_recall: %f" % (emo_val_bacc, emo_val_acc, emo_val_f1, emo_val_precision, emo_val_recall))

        return


def run(params,reader,logger):
    params = dataset.process_embedding(reader,params)###qnn/dataset/classification/__init__.py  def process_embedding
    print('\n print params ')
    print(params)
    qdnn = models.setup(params)
    model = qdnn.getModel() 
    print(model.summary())

    optimizer = units.getOptimizer(name=params.optimizer, lr=params.lr)
    earlystop = EarlyStopping(monitor='sar_val_acc', patience=35, mode='max', verbose=1)
    now = int(round(time.time()*1000))
    now = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(now/1000))
    board = TensorBoard(
        log_dir='./board/'+str(now),
        histogram_freq=2,
        batch_size=params.batch_size,
        write_graph=True,
        write_grads=True,
        write_images=True
        )
    model.compile(
        loss={'sarcasm-idenfity' : params.loss,
                'sentiment-analysis' : params.loss,
                'emotion-recognition': params.loss},
        loss_weights={
                'sarcasm-idenfity': 1/3,
                'sentiment-analysis': 1/3,
                'emotion-recognition': 1/3},
        optimizer=optimizer,
        metrics=['accuracy'])

    training_generator = TrainDataGenerator(datatype='train', batch_size=params.batch_size, shuffle=True)
    valdata = getValidData('dev')

    history = model.fit_generator(
        training_generator,
        epochs=params.epochs,
        validation_data=getXY('dev'),
        callbacks=[
            Metrics(valid_data=(valdata)),
            earlystop,
            board])
    evalresult = EvalModel(model=model)


    return history, evalresult

model_name = 'qdnn'
grid_parameters ={
        "dataset_name":["SST_2"],
        "wordvec_path":["glove/glove.6B.50d.txt"],#"glove/glove.6B.300d.txt"],"glove/normalized_vectors.txt","glove/glove.6B.50d.txt","glove/glove.6B.100d.txt",
        "loss": ["categorical_crossentropy"],#"mean_squared_error"],,"categorical_hinge"
        "optimizer":["adam"], #"adagrad","adamax","nadam"],,"adadelta","adam"    "rmsprop"-default
        "batch_size":[48],#16,32 64 
        "activation":["relu"],###relu sigmoid
        "amplitude_l2":[0.00000003], #0.0000005,0.0000001,0.00000001
        "phase_l2":[0.00000003],
        "dense_l2":[0.00000003],#0.0001,0.00001,0],
        "measurement_size" :[1000],#,50 100 300],
        "lr" : [0.0075],#,0.001 0.0001 0.01
        "epochs" : [20],
        "dropout_rate_embedding" : [0.2],#0.5,0.75,0.8,0.9,1],
        "dropout_rate_probs" : [0.2],#,0.5,0.75,0.8,1]    ,
        "ablation" : [1],
        "network_type" : [model_name]
    }
if __name__=="__main__":

    logging.basicConfig(filemode='w')
    logger = logging.getLogger(__name__)
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler("explog-test-"+model_name+".txt")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    now = int(round(time.time()*1000))
    now = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(now/1000))
    logger.info("\n\n***************"+str(now)+"************Start print log*********************************************")

    parser = argparse.ArgumentParser(description='running the complex embedding network')
    parser.add_argument('-gpu_num', action = 'store', dest = 'gpu_num', help = 'please enter the gpu num.',default=gpu_count)
    parser.add_argument('-gpu', action = 'store', dest = 'gpu', help = 'please enter the gpu num.',default=0)
    args = parser.parse_args()
    
    for index, arg in enumerate(itertools.product(*grid_parameters.values())):
        print(index)
        print(args.gpu_num)
        print(args.gpu)
        # print(index%args.gpu_num==args.gpu)
        print(arg)

    parameters= [arg for index,arg in enumerate(itertools.product(*grid_parameters.values())) if index%args.gpu_num==args.gpu]
     
    parameters= parameters[::-1]
    print('experiment/qnn/run_classification.py')
    print(parameters)

    logger.info("parameters: %s",parameters)

    params = Params()
    config_file = 'config/qdnn.ini'    # define dataset in the config
    params.parse_config(config_file)    
    for parameter in parameters:
        old_dataset = params.dataset_name
        params.setup(zip(grid_parameters.keys(),parameter))
        if old_dataset != params.dataset_name:
            print("switch {} to {}".format(old_dataset,params.dataset_name))
            reader=dataset.setup(params) 
            params.reader = reader 

        history, evalresult = run(params,reader,logger)
        logger.info("history: %s",str(history))

        logger.info("- eval_bacc: %f - eval_acc: %f  — eval_f1: %f — eval_precision: %f — eval_recall: %f — specificity: %f" % (evalresult[5], evalresult[6], evalresult[7], evalresult[8], evalresult[9], evalresult[10]))
        logger.info("- sent_eval_bacc: %f - sent_eval_acc: %f  — sent_eval_f1: %f — sent_eval_precision: %f — sent_eval_recall: %f" % (evalresult[0], evalresult[1], evalresult[2], evalresult[3],evalresult[4]))
        logger.info("- emo_eval_bacc: %f - emo_eval_acc: %f  — emo_eval_f1: %f — emo_eval_precision: %f — emo_eval_recall: %f" % (evalresult[11], evalresult[12], evalresult[13], evalresult[14],evalresult[15]))
        K.clear_session()

    logger.info("*********************************************Finish*********************************************")



