import keras
import math
import cv2
import numpy as np
import csv
# from efficientnet_pytorch import EfficientNet
# from PIL import Image
# from torchvision import transforms
import os
# import torchvggish.torchvggish.vggish_input as vggish_input
from keras.models import Sequential
from keras.layers import Dense,Flatten
import random
import time


def readcsv(file_name):
    with open(file_name, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        utt_namelist = []
        # context = np.array()
        for i in reader:
            if i[0] != '':
                utt_namelist.append(i[0])
        utt_namelist = list(set(utt_namelist))
        utt_namelist.remove("KEY")
        utt_dict = {}
        for name in utt_namelist:
            utt_dict[name] = {}
            utt_dict[name]['utterance'] = []
            utt_dict[name]['utt-number'] = ''
            utt_dict[name]['sentiment-label'] = []
            utt_dict[name]['emotion-label'] = []
            utt_dict[name]['sarcasm-label'] = []

    with open(file_name, 'r', encoding='utf-8') as file1:
        reader = csv.reader(file1)
        for item in reader:
            if item[0] == 'KEY' or item[0] == '':
                continue
            utt_dict[item[0]]['sarcasm-label'].append(item[4])
            utt_dict[item[0]]['sentiment-label'].append(item[5])
            utt_dict[item[0]]['emotion-label'].append(item[7])
            utt_dict[item[0]]['utterance'].append(item[2])
            utt_dict[item[0]]['utt-number'] = item[0]
    return utt_dict, utt_namelist


def process_utt_dict(utt_dict):
    for key in utt_dict.keys():
        len_utt = len(utt_dict[key]['utterance'])
        if len_utt == 2:
            utt_dict[key]['utterance'].insert(0, '')
            utt_dict[key]['utterance'].insert(0, '')

            utt_dict[key]['sarcasm-label'].insert(0, 'False')
            utt_dict[key]['sarcasm-label'].insert(0, 'False')
            utt_dict[key]['sentiment-label'].insert(0, 0)
            utt_dict[key]['sentiment-label'].insert(0, 0)
            utt_dict[key]['emotion-label'].insert(0, 7)
            utt_dict[key]['emotion-label'].insert(0, 7)
        elif len_utt >= 4:
            utt_dict[key]['utterance'] = utt_dict[key]['utterance'][-4:]
            utt_dict[key]['sarcasm-label'] = utt_dict[key]['sarcasm-label'][-4:]
            utt_dict[key]['sentiment-label'] = utt_dict[key]['sentiment-label'][-4:]
            utt_dict[key]['emotion-label'] = utt_dict[key]['emotion-label'][-4:]
        elif len_utt == 3:
            utt_dict[key]['utterance'].insert(0, '')
            utt_dict[key]['sarcasm-label'].insert(0, 'False')
            utt_dict[key]['sentiment-label'].insert(0, 0)
            utt_dict[key]['emotion-label'].insert(0, 7)
        else:
            continue
    return utt_dict

def getmeldweight(batch_size):
        weight = [1]
        allWeights = []
        for i in range(batch_size):
            allWeights.append(weight)
        allWeights = np.array(allWeights)
        return allWeights


class TrainDataGenerator(keras.utils.Sequence):

    def __init__(self, datatype, batch_size, shuffle=True):
        super().__init__()
        self.datatype = datatype
        self.batch_size = batch_size
        if self.datatype == 'train':
            datafile = '../../dataset/meld/meld-dataset-train.csv'
            self.datalen = 500
        elif self.datatype == 'dev':
            datafile = '../../dataset/meld/meld-dataset-dev.csv'
            self.datalen = 90
        elif self.datatype == 'test':
            datafile = '../../dataset/meld/meld-dataset-test.csv'
            self.datalen = 100
        uttDict, self.uttNameList = readcsv(datafile)
        uttDict = process_utt_dict(uttDict)
        self.uttList = list(uttDict.values())

        self.frameNumbers = 4
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        print('math.ceil(self.datalen / self.batch_size):', math.ceil(self.datalen / self.batch_size))
        return int(math.ceil(self.datalen / self.batch_size))


    def __getitem__(self, index):
        batch_indexs =\
            self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        text_amp_list = []
        text_phase_list = []
        image_amp_list = []
        image_phase_list = []
        audio_amp_list = []
        audio_phase_list = []
        sar_label_list = []
        sent_label_list = []
        emo_label_list = []
        for i in batch_indexs:
            uttName = self.uttList[i]['utt-number']
            textPath = \
            "../../dataset/meld/textFeature/"+self.datatype+"/"+uttName+"/"
            textFea = [np.loadtxt(
                            textPath+str(num),
                            dtype=float,
                            delimiter=",") for num in range(4)]
            textFea = [np.expand_dims(item, axis=0) for item in textFea]
            textFea = np.concatenate(textFea, axis=0)
            text_amp_list.append(textFea)

            imagePath = \
                "../../dataset/meld/imageFeature/"+self.datatype+"/"+uttName+"/"
            imageFea = [np.loadtxt(
                            imagePath+str(num),
                            dtype=float,
                            delimiter=",") for num in range(4)]
            imageFea = [np.expand_dims(item, axis=0) for item in imageFea]
            imageFea = np.concatenate(imageFea, axis=0)
            image_amp_list.append(imageFea)

            audioPath = "../../dataset/meld/audioFeature/"+self.datatype+"/"
            tAudioFea = audioPath+"target/"+uttName
            cAudioFea = audioPath+"context/"+uttName
            tAudioFea = np.expand_dims(
                            np.loadtxt(\
                                tAudioFea, \
                                dtype=float, \
                                delimiter=","), axis=0)
            tAudioFea = np.expand_dims(tAudioFea, axis=1)
            cAudioFea = np.loadtxt(
                            cAudioFea,
                            dtype=float,
                            delimiter=",")
            cAudioFea = np.expand_dims(cAudioFea, axis=1)
            audioFea = np.array(
                np.concatenate([cAudioFea, tAudioFea], axis=0).squeeze())
            audio_amp_list.append(audioFea)
            sarcasm_list = self.uttList[i]['sarcasm-label']
            sentiment_list = self.uttList[i]['sentiment-label']
            emotion_list = self.uttList[i]['emotion-label']
            sar_label = []
            sent_label = []
            emo_label = []
            text_phase = []
            image_phase = []
            audio_phase = []
            for sar in sarcasm_list:
                if 'True' == sar:
                    sarcasmLabel = np.array([0, 1], dtype=np.int8)
                else:
                    sarcasmLabel = np.array([1, 0], dtype=np.int8)
                sar_label.append(sarcasmLabel)
            sar_label_list.append(sar_label)

            for sent in sentiment_list:
                sentimentLabel = np.zeros(3, dtype=np.int8)
                if -1 == int(sent):
                    sentimentLabel[0] = 1
                    text_phase.append(
                        np.random.uniform(
                            low=0.0,
                            high=(2/3)*math.pi, size=(128)))
                    image_phase.append(
                        np.random.uniform(
                            low=0.0,
                            high=(2/3)*math.pi, size=(128)))
                    audio_phase.append(
                        np.random.uniform(
                            low=0.0,
                            high=(2/3)*math.pi, size=(128)))
                elif 0 == int(sent):
                    sentimentLabel[1] = 1
                    text_phase.append(
                        np.random.uniform(
                            low=(2/3)*math.pi,
                            high=(4/3)*math.pi, size=(128)))
                    image_phase.append(
                        np.random.uniform(
                            low=(2/3)*math.pi,
                            high=(4/3)*math.pi, size=(128)))
                    audio_phase.append(
                        np.random.uniform(
                            low=(2/3)*math.pi,
                            high=(4/3)*math.pi, size=(128)))
                else:
                    sentimentLabel[2] = 1
                    text_phase.append(
                        np.random.uniform(
                            low=(4/3)*math.pi,
                            high=2*math.pi, size=(128)))
                    audio_phase.append(
                        np.random.uniform(
                            low=(4/3)*math.pi,
                            high=2*math.pi, size=(128)))
                    image_phase.append(
                        np.random.uniform(
                            low=(4/3)*math.pi,
                            high=2*math.pi, size=(128)))
                sent_label.append(sentimentLabel)
            sent_label_list.append(sent_label)

            for emo in emotion_list:
                emotionLabel = np.zeros(9, dtype=np.int8)
                if len(str(emo).split(',')) != 0:
                    emotionLabel[int(str(emo).split(',')[0])-1] = 1
                else:
                    emotionLabel[int(emo)-1] = 1
                emo_label.append(emotionLabel)
            emo_label_list.append(emo_label)

            text_phase_list.append(text_phase)
            image_phase_list.append(image_phase)
            audio_phase_list.append(audio_phase)

        sar_label_list = np.array(sar_label_list)
        sent_label_list = np.array(sent_label_list)
        emo_label_list = np.array(emo_label_list)
        weight_out = getmeldweight(len(batch_indexs))
        text_amp_list = np.array(text_amp_list)
        image_amp_list = np.array(image_amp_list)
        audio_amp_list = np.array(audio_amp_list)

        text_phase_list = np.array(text_phase_list)
        image_phase_list = np.array(image_phase_list)
        audio_phase_list = np.array(audio_phase_list)
        x_batch = {'text_amp': text_amp_list,
                   'image_amp': image_amp_list,
                   'audio_amp': audio_amp_list,
                   'text_phase': text_phase_list,
                   'image_phase': image_phase_list,
                   'audio_phase': audio_phase_list,
                   'weight': weight_out}
        y_batch = {'sarcasm-idenfity': sar_label_list[:, -1, :].squeeze(),
                   'sentiment-analysis': sent_label_list[:, -1, :].squeeze(),
                   'emotion-recognition': emo_label_list[:, -1, :].squeeze()}
        
        return x_batch, y_batch

    def on_epoch_end(self):
        self.indexes = np.arange(self.datalen)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

class TrainDataGenerator_double_real(keras.utils.Sequence):

    def __init__(self, datatype, batch_size, shuffle=True):
        super().__init__()
        self.datatype = datatype
        self.batch_size = batch_size
        if self.datatype == 'train':
            datafile = '../../dataset/meld/meld-dataset-train.csv'
            self.datalen = 500
        elif self.datatype == 'dev':
            datafile = '../../dataset/meld/meld-dataset-dev.csv'
            self.datalen = 90
        elif self.datatype == 'test':
            datafile = '../../dataset/meld/meld-dataset-test.csv'
            self.datalen = 100
        uttDict, self.uttNameList = readcsv(datafile)
        uttDict = process_utt_dict(uttDict)
        self.uttList = list(uttDict.values())

        self.frameNumbers = 4
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        print('math.ceil(self.datalen / self.batch_size):', math.ceil(self.datalen / self.batch_size))
        return int(math.ceil(self.datalen / self.batch_size))


    def __getitem__(self, index):
        batch_indexs =\
            self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        text_amp_list = []
        text_phase_list = []
        image_amp_list = []
        image_phase_list = []
        audio_amp_list = []
        audio_phase_list = []
        sar_label_list = []
        sent_label_list = []
        emo_label_list = []
        for i in batch_indexs:
            uttName = self.uttList[i]['utt-number']
            textPath = \
            "../../dataset/meld/textFeature/"+self.datatype+"/"+uttName+"/"
            textFea = [np.loadtxt(
                            textPath+str(num),
                            dtype=float,
                            delimiter=",") for num in range(4)]
            textFea = [np.expand_dims(item, axis=0) for item in textFea]
            textFea = np.concatenate(textFea, axis=0)
            text_amp_list.append(textFea)
            imagePath = \
                "../../dataset/meld/imageFeature/"+self.datatype+"/"+uttName+"/"
            imageFea = [np.loadtxt(
                            imagePath+str(num),
                            dtype=float,
                            delimiter=",") for num in range(4)]
            imageFea = [np.expand_dims(item, axis=0) for item in imageFea]
            imageFea = np.concatenate(imageFea, axis=0)
            image_amp_list.append(imageFea)
            # torch.Size([4, 3, 480, 360])
            audioPath = "../../dataset/meld/audioFeature/"+self.datatype+"/"
            tAudioFea = audioPath+"target/"+uttName
            cAudioFea = audioPath+"context/"+uttName
            tAudioFea = np.expand_dims(
                            np.loadtxt(\
                                tAudioFea, \
                                dtype=float, \
                                delimiter=","), axis=0)
            tAudioFea = np.expand_dims(tAudioFea, axis=1)
            cAudioFea = np.loadtxt(
                            cAudioFea,
                            dtype=float,
                            delimiter=",")
            cAudioFea = np.expand_dims(cAudioFea, axis=1)
            audioFea = np.array(
                np.concatenate([cAudioFea, tAudioFea], axis=0).squeeze())
            audio_amp_list.append(audioFea)
            sarcasm_list = self.uttList[i]['sarcasm-label']
            sentiment_list = self.uttList[i]['sentiment-label']
            emotion_list = self.uttList[i]['emotion-label']
            sar_label = []
            sent_label = []
            emo_label = []
            text_phase = []
            image_phase = []
            audio_phase = []
            for sar in sarcasm_list:
                if 'True' == sar:
                    sarcasmLabel = np.array([0, 1], dtype=np.int8)
                else:
                    sarcasmLabel = np.array([1, 0], dtype=np.int8)
                sar_label.append(sarcasmLabel)
            sar_label_list.append(sar_label)

            for sent in sentiment_list:
                sentimentLabel = np.zeros(3, dtype=np.int8)
                if -1 == int(sent):
                    sentimentLabel[0] = 1
                    text_phase.append(
                        np.random.uniform(
                            low=0.0,
                            high=(2/3)*math.pi, size=(128)))
                    image_phase.append(
                        np.random.uniform(
                            low=0.0,
                            high=(2/3)*math.pi, size=(128)))
                    audio_phase.append(
                        np.random.uniform(
                            low=0.0,
                            high=(2/3)*math.pi, size=(128)))
                elif 0 == int(sent):
                    sentimentLabel[1] = 1
                    text_phase.append(
                        np.random.uniform(
                            low=(2/3)*math.pi,
                            high=(4/3)*math.pi, size=(128)))
                    image_phase.append(
                        np.random.uniform(
                            low=(2/3)*math.pi,
                            high=(4/3)*math.pi, size=(128)))
                    audio_phase.append(
                        np.random.uniform(
                            low=(2/3)*math.pi,
                            high=(4/3)*math.pi, size=(128)))
                else:
                    sentimentLabel[2] = 1
                    text_phase.append(
                        np.random.uniform(
                            low=(4/3)*math.pi,
                            high=2*math.pi, size=(128)))
                    audio_phase.append(
                        np.random.uniform(
                            low=(4/3)*math.pi,
                            high=2*math.pi, size=(128)))
                    image_phase.append(
                        np.random.uniform(
                            low=(4/3)*math.pi,
                            high=2*math.pi, size=(128)))
                sent_label.append(sentimentLabel)
            sent_label_list.append(sent_label)

            for emo in emotion_list:
                emotionLabel = np.zeros(9, dtype=np.int8)
                if len(str(emo).split(',')) != 0:
                    emotionLabel[int(str(emo).split(',')[0])-1] = 1
                else:
                    emotionLabel[int(emo)-1] = 1
                emo_label.append(emotionLabel)
            emo_label_list.append(emo_label)

            text_phase_list.append(text_phase)
            image_phase_list.append(image_phase)
            audio_phase_list.append(audio_phase)

        sar_label_list = np.array(sar_label_list)
        sent_label_list = np.array(sent_label_list)
        emo_label_list = np.array(emo_label_list)
        weight_out = getmeldweight(len(batch_indexs))
        text_amp_list = np.array(text_amp_list)
        image_amp_list = np.array(image_amp_list)
        audio_amp_list = np.array(audio_amp_list)

        text_phase_list = np.array(text_phase_list)
        image_phase_list = np.array(image_phase_list)
        audio_phase_list = np.array(audio_phase_list)
        x_batch = {'text_amp': text_amp_list,
                   'image_amp': image_amp_list,
                   'audio_amp': audio_amp_list,
                   'text_phase': text_amp_list,
                   'image_phase': image_amp_list,
                   'audio_phase': audio_amp_list,
                   'weight': weight_out}
        y_batch = {'sarcasm-idenfity': sar_label_list[:, -1, :].squeeze(),
                   'sentiment-analysis': sent_label_list[:, -1, :].squeeze(),
                   'emotion-recognition': emo_label_list[:, -1, :].squeeze()}
        return x_batch, y_batch

    def on_epoch_end(self):
        self.indexes = np.arange(self.datalen)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)



def getValidData(datatype):
    if datatype == 'dev':
        audioPath = "../../dataset/meld/audioFeature/dev/"
        textPath = "../../dataset/meld/textFeature/dev/"
        imagePath = "../../dataset/meld/imageFeature/dev/"
        vallen = 90
        datafile = '../../dataset/meld/meld-dataset-dev.csv'
        uttDict, uttNameList = readcsv(datafile)
        uttDict = process_utt_dict(uttDict)
    else:
        audioPath = "../../dataset/meld/audioFeature/test/"
        textPath = "../../dataset/meld/textFeature/test/"
        imagePath = "../../dataset/meld/imageFeature/test/"
        vallen = 100
        datafile = '../../dataset/meld/meld-dataset-test.csv'
        uttDict, uttNameList = readcsv(datafile)
        uttDict = process_utt_dict(uttDict)

    def gen_ling_amp_pha_sarl_sentl(textPath, imagePath, audioPath, uttDict):
        textFeaList = []
        imageFeaList = []
        tAudioList = []
        cAudioList = []
        sarList = []
        sentList = []
        emoList = []
        text_phase_list = []
        image_phase_list = []
        audio_phase_list = []
        count = 0
        for name in uttNameList:
            _textPath = textPath + name+"/"
            textFea = [np.loadtxt(
                _textPath+str(num),
                dtype=float,
                delimiter=",") for num in range(4)]
            textFea = [np.expand_dims(item, axis=0) for item in textFea]
            textFea = np.concatenate(textFea, axis=0)
            textFeaList.append(textFea)
            _imagePath = imagePath + name+"/"
            imageFea = [np.loadtxt(
                _imagePath+str(num),
                dtype=float,
                delimiter=",") for num in range(4)]
            imageFea = [np.expand_dims(item, axis=0) for item in imageFea]
            imageFea = np.concatenate(imageFea, axis=0)
            imageFeaList.append(imageFea)
            _tAudioFea = audioPath+"target/"+name
            _cAudioFea = audioPath+"context/"+name
            tAudioFea = np.expand_dims(
                np.loadtxt(
                    _tAudioFea,
                    dtype=float,
                    delimiter=","), axis=0)
            tAudioFea = np.expand_dims(tAudioFea, axis=1)
            cAudioFea = np.loadtxt(
                _cAudioFea,
                dtype=float,
                delimiter=",")
            cAudioFea = np.expand_dims(cAudioFea, axis=1)
            tAudioList.append(tAudioFea)
            cAudioList.append(cAudioFea)
            sarcasm_list = uttDict[name]['sarcasm-label']
            sentiment_list = uttDict[name]['sentiment-label']
            emotion_list = uttDict[name]['emotion-label']
            sar_label = []
            sent_label = []
            emo_label = []
            text_phase = []
            image_phase = []
            audio_phase = []
            for sar in sarcasm_list:
                if 'True' == sar:
                    sarcasmLabel = np.array([0, 1], dtype=np.int8)
                else:
                    sarcasmLabel = np.array([1, 0], dtype=np.int8)
                sar_label.append(sarcasmLabel)
            sarList.append(sar_label)

            for sent in sentiment_list:
                sentimentLabel = np.zeros(3, dtype=np.int8)
                if -1 == int(sent):
                    sentimentLabel[0] = 1
                    text_phase.append(
                        np.random.uniform(
                            low=0.0,
                            high=(2/3)*math.pi, size=(128)))
                    image_phase.append(
                        np.random.uniform(
                            low=0.0,
                            high=(2/3)*math.pi, size=(128)))
                    audio_phase.append(
                        np.random.uniform(
                            low=0.0,
                         high=(2/3)*math.pi, size=(128)))
                elif 0 == int(sent):
                    sentimentLabel[1] = 1
                    text_phase.append(
                        np.random.uniform(
                            low=(2/3)*math.pi,
                            high=(4/3)*math.pi, size=(128)))
                    image_phase.append(
                        np.random.uniform(
                            low=(2/3)*math.pi,
                            high=(4/3)*math.pi, size=(128)))
                    audio_phase.append(
                        np.random.uniform(
                            low=(2/3)*math.pi,
                            high=(4/3)*math.pi, size=(128)))
                else:
                    sentimentLabel[2] = 1
                    text_phase.append(
                        np.random.uniform(
                            low=(4/3)*math.pi,
                            high=2*math.pi, size=(128)))
                    audio_phase.append(
                        np.random.uniform(
                            low=(4/3)*math.pi,
                            high=2*math.pi, size=(128)))
                    image_phase.append(
                        np.random.uniform(
                            low=(4/3)*math.pi,
                         high=2*math.pi, size=(128)))
                sent_label.append(sentimentLabel)
            sentList.append(sent_label)
            text_phase_list.append(text_phase)
            image_phase_list.append(image_phase)
            audio_phase_list.append(audio_phase)

            for emo in emotion_list:
                emotionLabel = np.zeros(9, dtype=np.int8)
                if len(str(emo).split(',')) != 0:
                    emotionLabel[int(str(emo).split(',')[0])-1] = 1
                else:
                    emotionLabel[int(emo)-1] = 1
                emo_label.append(emotionLabel)
            emoList.append(emo_label)
        textFeaList = np.array(textFeaList)
        imageFeaList = np.array(imageFeaList)
        tAudioList = np.array(tAudioList)
        cAudioList = np.array(cAudioList)
        sarList = np.array(sarList)
        sentList = np.array(sentList)
        emoList = np.array(emoList)
        text_phase_list = np.array(text_phase_list)
        image_phase_list = np.array(image_phase_list)
        audio_phase_list = np.array(audio_phase_list)
        return textFeaList, imageFeaList, tAudioList, cAudioList,\
            text_phase_list, image_phase_list, audio_phase_list,\
            sarList, sentList, emoList

    def getmeldweight():
        weight = [1]
        allWeights = []
        for i in range(vallen):
            allWeights.append(weight)
        allWeights = np.array(allWeights)
        return allWeights

    textFeaList, imageFeaList, tAudioList, cAudioList,\
        text_phase_list, image_phase_list, audio_phase_list,\
        sarList, sentList, emoList = \
        gen_ling_amp_pha_sarl_sentl(textPath, imagePath, audioPath, uttDict)
    weight = getmeldweight()
    if datatype == 'test':
        return textFeaList, imageFeaList,\
            np.concatenate([cAudioList, tAudioList], axis=1).squeeze(),\
            text_phase_list, image_phase_list, audio_phase_list,\
            sarList[:, -1, :].squeeze(),\
            sentList[:, -1, :].squeeze(),\
            emoList[:, -1, :].squeeze(),\
            weight, uttNameList
    else:
        return textFeaList, imageFeaList,\
            np.concatenate([cAudioList, tAudioList], axis=1).squeeze(),\
            text_phase_list, image_phase_list, audio_phase_list,\
            sarList[:, -1, :].squeeze(),\
            sentList[:, -1, :].squeeze(),\
            emoList[:, -1, :].squeeze(),\
            weight

def getValidData_double_real(datatype):
    if datatype == 'dev':
        audioPath = "../../dataset/meld/audioFeature/dev/"
        textPath = "../../dataset/meld/textFeature/dev/"
        imagePath = "../../dataset/meld/imageFeature/dev/"
        vallen = 90
        datafile = '../../dataset/meld/meld-dataset-dev.csv'
        uttDict, uttNameList = readcsv(datafile)
        uttDict = process_utt_dict(uttDict)
    else:
        audioPath = "../../dataset/meld/audioFeature/test/"
        textPath = "../../dataset/meld/textFeature/test/"
        imagePath = "../../dataset/meld/imageFeature/test/"
        vallen = 100
        datafile = '../../dataset/meld/meld-dataset-test.csv'
        uttDict, uttNameList = readcsv(datafile)
        uttDict = process_utt_dict(uttDict)

    def gen_ling_amp_pha_sarl_sentl(textPath, imagePath, audioPath, uttDict):
        textFeaList = []
        imageFeaList = []
        tAudioList = []
        cAudioList = []
        sarList = []
        sentList = []
        emoList = []
        text_phase_list = []
        image_phase_list = []
        audio_phase_list = []
        count = 0
        for name in uttNameList:
            '''
            从文件夹中读取文本特征数据
            '''
            _textPath = textPath + name+"/"
            textFea = [np.loadtxt(
                _textPath+str(num),
                dtype=float,
                delimiter=",") for num in range(4)]
            textFea = [np.expand_dims(item, axis=0) for item in textFea]
            textFea = np.concatenate(textFea, axis=0)
            textFeaList.append(textFea)

            _imagePath = imagePath + name+"/"
            imageFea = [np.loadtxt(
                _imagePath+str(num),
                dtype=float,
                delimiter=",") for num in range(4)]
            imageFea = [np.expand_dims(item, axis=0) for item in imageFea]
            imageFea = np.concatenate(imageFea, axis=0)
            imageFeaList.append(imageFea)

            _tAudioFea = audioPath+"target/"+name
            _cAudioFea = audioPath+"context/"+name
            tAudioFea = np.expand_dims(
                np.loadtxt(
                    _tAudioFea,
                    dtype=float,
                    delimiter=","), axis=0)
            tAudioFea = np.expand_dims(tAudioFea, axis=1)
            cAudioFea = np.loadtxt(
                _cAudioFea,
                dtype=float,
                delimiter=",")
            cAudioFea = np.expand_dims(cAudioFea, axis=1)
            tAudioList.append(tAudioFea)
            cAudioList.append(cAudioFea)

            sarcasm_list = uttDict[name]['sarcasm-label']
            sentiment_list = uttDict[name]['sentiment-label']
            emotion_list = uttDict[name]['emotion-label']
            sar_label = []
            sent_label = []
            emo_label = []
            text_phase = []
            image_phase = []
            audio_phase = []
            for sar in sarcasm_list:
                if 'True' == sar:
                    sarcasmLabel = np.array([0, 1], dtype=np.int8)
                else:
                    sarcasmLabel = np.array([1, 0], dtype=np.int8)
                sar_label.append(sarcasmLabel)
            sarList.append(sar_label)

            for sent in sentiment_list:
                sentimentLabel = np.zeros(3, dtype=np.int8)
                if -1 == int(sent):
                    sentimentLabel[0] = 1
                    text_phase.append(
                        np.random.uniform(
                            low=0.0,
                            high=(2/3)*math.pi, size=(128)))
                    image_phase.append(
                        np.random.uniform(
                            low=0.0,
                            high=(2/3)*math.pi, size=(128)))
                    audio_phase.append(
                        np.random.uniform(
                            low=0.0,
                         high=(2/3)*math.pi, size=(128)))
                elif 0 == int(sent):
                    sentimentLabel[1] = 1
                    text_phase.append(
                        np.random.uniform(
                            low=(2/3)*math.pi,
                            high=(4/3)*math.pi, size=(128)))
                    image_phase.append(
                        np.random.uniform(
                            low=(2/3)*math.pi,
                            high=(4/3)*math.pi, size=(128)))
                    audio_phase.append(
                        np.random.uniform(
                            low=(2/3)*math.pi,
                            high=(4/3)*math.pi, size=(128)))
                else:
                    sentimentLabel[2] = 1
                    text_phase.append(
                        np.random.uniform(
                            low=(4/3)*math.pi,
                            high=2*math.pi, size=(128)))
                    audio_phase.append(
                        np.random.uniform(
                            low=(4/3)*math.pi,
                            high=2*math.pi, size=(128)))
                    image_phase.append(
                        np.random.uniform(
                            low=(4/3)*math.pi,
                         high=2*math.pi, size=(128)))
                sent_label.append(sentimentLabel)
            sentList.append(sent_label)
            text_phase_list.append(text_phase)
            image_phase_list.append(image_phase)
            audio_phase_list.append(audio_phase)

            for emo in emotion_list:
                emotionLabel = np.zeros(9, dtype=np.int8)
                if len(str(emo).split(',')) != 0:
                    emotionLabel[int(str(emo).split(',')[0])-1] = 1
                else:
                    emotionLabel[int(emo)-1] = 1
                emo_label.append(emotionLabel)
            emoList.append(emo_label)
        textFeaList = np.array(textFeaList)
        imageFeaList = np.array(imageFeaList)
        tAudioList = np.array(tAudioList)
        cAudioList = np.array(cAudioList)
        sarList = np.array(sarList)
        sentList = np.array(sentList)
        emoList = np.array(emoList)
        text_phase_list = np.array(text_phase_list)
        image_phase_list = np.array(image_phase_list)
        audio_phase_list = np.array(audio_phase_list)
        return textFeaList, imageFeaList, tAudioList, cAudioList,\
            text_phase_list, image_phase_list, audio_phase_list,\
            sarList, sentList, emoList

    def getmeldweight():
        weight = [1]
        allWeights = []
        for i in range(vallen):
            allWeights.append(weight)
        allWeights = np.array(allWeights)
        return allWeights

    textFeaList, imageFeaList, tAudioList, cAudioList,\
        text_phase_list, image_phase_list, audio_phase_list,\
        sarList, sentList, emoList = \
        gen_ling_amp_pha_sarl_sentl(textPath, imagePath, audioPath, uttDict)
    weight = getmeldweight()

    if datatype == 'test':
        return textFeaList, imageFeaList,\
            np.concatenate([cAudioList, tAudioList], axis=1).squeeze(),\
            textFeaList, imageFeaList, np.concatenate([cAudioList, tAudioList], axis=1).squeeze(),\
            sarList[:, -1, :].squeeze(),\
            sentList[:, -1, :].squeeze(),\
            emoList[:, -1, :].squeeze(),\
            weight, uttNameList
    else:
        return textFeaList, imageFeaList,\
            np.concatenate([cAudioList, tAudioList], axis=1).squeeze(),\
            textFeaList, imageFeaList, np.concatenate([cAudioList, tAudioList], axis=1).squeeze(),\
            sarList[:, -1, :].squeeze(),\
            sentList[:, -1, :].squeeze(),\
            emoList[:, -1, :].squeeze(),\
            weight      

def getXY(datatype):
    textFeaList, imageFeaList, audioFeaList,\
       textPhaseList, imagePhaseList, audioPhaseList,\
           sarList, sentList, emoList, weight = getValidData(datatype)
    x = {
        'text_amp':textFeaList,
        'image_amp':imageFeaList,
        'audio_amp':audioFeaList,
        'text_phase':textPhaseList,
        'image_phase':imagePhaseList,
        'audio_phase':audioPhaseList,
        'weight':weight
    }
    y = {
        'sarcasm-idenfity':sarList,
        'sentiment-analysis':sentList,
        'emotion-recognition':emoList
    }
    return x, y

def getXY_double_real(datatype):
    textFeaList, imageFeaList, audioFeaList,\
       textPhaseList, imagePhaseList, audioPhaseList,\
           sarList, sentList, emoList, weight = getValidData(datatype)
    x = {
        'text_amp':textFeaList,
        'image_amp':imageFeaList,
        'audio_amp':audioFeaList,
        'text_phase':textFeaList,
        'image_phase':imageFeaList,
        'audio_phase':audioFeaList,
        'weight':weight
    }
    y = {
        'sarcasm-idenfity':sarList,
        'sentiment-analysis':sentList,
        'emotion-recognition':emoList
    }
    return x, y





if __name__ == '__main__':

    textFeaList, imageFeaList, audioFeaList,\
        textPhaseList, imagePhaseList, audioPhaseList,\
        sarList, sentList, emoList, weight = \
        getValidData(datatype='dev')



