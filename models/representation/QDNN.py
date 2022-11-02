
# -*- coding: utf-8 -*-
from os import name
from keras.layers.merge import concatenate
import tensorflow as tf
from keras.layers import Layer,Embedding, GlobalAveragePooling1D, Dense, Masking, Flatten,Dropout, Activation,Multiply,Concatenate,Add,Subtract,Reshape,LeakyReLU,Lambda
from keras.layers import GRU
from tensorflow.python.ops.gen_array_ops import shape
from models.BasicModel import BasicModel
from keras.models import Model, Input, model_from_json, load_model
from keras.constraints import unit_norm
from layers import *
#import keras.backend as K
from keras import backend as K
import math
import numpy as np
from tensorflow.keras import regularizers


class QDNN(BasicModel):

    def initialize(self):
        self.text_amp =\
            Input(shape=(4, 768,), dtype='float32', name='text_amp')
        self.image_amp =\
            Input(shape=(4, 2048,), dtype='float32', name='image_amp')
        self.audio_amp =\
            Input(shape=(4, 128,), dtype='float32', name='audio_amp')

        self.text_phase =\
            Input(shape=(4, 128,), dtype='float32', name='text_phase')
        self.image_phase =\
            Input(shape=(4, 128,), dtype='float32', name='image_phase')
        self.audio_phase =\
            Input(shape=(4, 128,), dtype='float32', name='audio_phase')
        # 当做weight，自己定义的weight
        self.weight = Input(shape=(1,), dtype='float32', name='weight')

        self.sardense = Dense(
            2, activation='softmax',
            kernel_regularizer=regularizers.l2(self.opt.dense_l2),
            name='sarcasm-idenfity',
            kernel_initializer='glorot_normal'
            )  # activation="sigmoid",
        self.sentdense = Dense(
            3, activation='softmax',
            kernel_regularizer=regularizers.l2(self.opt.dense_l2),
            name='sentiment-analysis',
            kernel_initializer='glorot_normal'
            )
        self.emodense = Dense(
            9, activation='softmax',
            kernel_regularizer=regularizers.l2(self.opt.dense_l2),
            name='emotion-recognition',
            kernel_initializer='glorot_normal'
            )
        self.dropout_embedding = Dropout(
            self.opt.dropout_rate_embedding,
            name='dropout-embedding')
        self.dropout_probs = Dropout(
            self.opt.dropout_rate_probs,
            name='dropout-probs')
        self.projection_task1 =\
            ComplexMeasurement(units=self.opt.measurement_size,
            name='projection-task1')
        self.projection_task2 =\
            ComplexMeasurement(units=self.opt.measurement_size,
            name='projection-task2')
        self.projection_task3 =\
            ComplexMeasurement(units=self.opt.measurement_size,
            name='projection-task3')

    def __init__(self,opt):
        super(QDNN, self).__init__(opt)

    def build(self):
        # #####下面写好方法，在这里构建

        probs1, probs2, prob3 = self.get_representation(
            self.text_amp,
            self.image_amp,
            self.audio_amp,
            self.text_phase,
            self.image_phase,
            self.audio_phase,
            self.weight
            )
        sentoutput = self.sentdense(probs1)
        saroutput = self.sardense(probs2)
        emooutput = self.emodense(prob3)

        # model = Model([self.input_amplitude,self.input_phase,self.weight,self.visual_amplitude,self.visual_phase], [output,sentoutput])
        model = Model([
            self.text_amp,
            self.image_amp,
            self.audio_amp,
            self.text_phase,
            self.image_phase,
            self.audio_phase,
            self.weight],
            [sentoutput, saroutput, emooutput])
        # model = Model([self.input_amplitude,self.input_phase,self.weight,self.visual_amplitude,self.visual_phase], [output])
        return model

    def get_representation(
            self, text_amp, image_amp, audio_amp,
            phase_text, phase_image, phase_audio, weight):
        self.weight = weight
        self.amplitude_text = text_amp
        self.amplitude_image = image_amp
        self.amplitude_audio = audio_amp
        self.phase_text = phase_text
        self.phase_image = phase_image
        self.phase_audio = phase_audio
        scale_size = 128  # #将要用于张量积的词向量的全连接层的维度

        text_amp_hidden_state = GRU(
            units=scale_size,
            return_state=True,
            return_sequences=True,
            name='text-amp-hidden-state')(self.amplitude_text)
        image_amp_hidden_state = GRU(
            units=scale_size,
            return_state=True,
            return_sequences=True,
            name='image-amp-hidden-state')(self.amplitude_image)
        audio_amp_hidden_state = GRU(
            units=scale_size,
            return_state=True,
            return_sequences=True,
            name='audio-amp-hidden-state')(self.amplitude_audio)

        text_phase_hidden_state = GRU(
            units=scale_size,
            return_state=True,
            return_sequences=True,
            name='text-phase-hidden-state')(self.phase_text)
        image_phase_hidden_state = GRU(
            units=scale_size,
            return_state=True,
            return_sequences=True,
            name='image-phase-hidden-state')(self.phase_image)
        audio_phase_hidden_state = GRU(
            units=scale_size,
            return_state=True,
            return_sequences=True,
            name='audio-phase-hidden-state')(self.phase_audio)

        text_cos = Lambda(
            lambda x:  K.cos(x), name='text-cos-i')(text_phase_hidden_state[0])
        text_sin = Lambda(
            lambda x:  K.sin(x), name='text-sin-i')(text_phase_hidden_state[0])
        text_real = Multiply(
            name='text-real')([text_cos, text_amp_hidden_state[0]])
        text_imag = Multiply(
            name='text-imag')([text_sin, text_amp_hidden_state[0]])
        [text_emb_r, text_emb_i] =\
            ComplexMixture()([text_real, text_imag, self.weight])

        image_cos = Lambda(
            lambda x:  K.cos(x), name='image-cos-i')(image_phase_hidden_state[0])
        image_sin = Lambda(
            lambda x:  K.sin(x), name='image-sin-i')(image_phase_hidden_state[0])
        image_real = Multiply(
            name='image-real')([image_cos, image_amp_hidden_state[0]])
        image_imag = Multiply(
            name='image-imag')([image_sin, image_amp_hidden_state[0]])
        [image_emb_r, image_emb_i] =\
            ComplexMixture()([image_real, image_imag, self.weight])

        audio_cos = Lambda(
            lambda x:  K.cos(x), name='audio-cos-i')(audio_phase_hidden_state[0])
        audio_sin = Lambda(
            lambda x:  K.sin(x), name='audio-sin-i')(audio_phase_hidden_state[0])
        audio_real = Multiply(
            name='audio-real')([audio_cos, audio_amp_hidden_state[0]])
        audio_imag = Multiply(
            name='audio-imag')([audio_sin, audio_amp_hidden_state[0]])
        [audio_emb_r, audio_emb_i] =\
            ComplexMixture()([audio_real, audio_imag, self.weight])

        text_image_real = Lambda(
            lambda x: 0.5*x[0]+0.5*x[1]+K.sqrt(x[0]*x[1]+1e-7),
            name='text-image-r')\
                ([text_emb_r, image_emb_r])
        text_audio_real = Lambda(
            lambda x: 0.5*x[0]+0.5*x[1]+K.sqrt(x[0]*x[1]+1e-7),
            name='text-audio-r')\
                ([text_emb_r, audio_emb_r])
        audio_image_real = Lambda(
            lambda x: 0.5*x[0]+0.5*x[1]+K.sqrt(x[0]*x[1]+1e-7),
            name='audio-image-r')\
                ([audio_emb_r, image_emb_r])
        text_image_imag = Lambda(
            lambda x: 0.5*x[0]+0.5*x[1]+K.sqrt(x[0]*x[1]+1e-7),
            name='text-image-i')\
                ([text_emb_i, image_emb_i])
        text_audio_imag = Lambda(
            lambda x: 0.5*x[0]+0.5*x[1]+K.sqrt(x[0]*x[1]+1e-7),
            name='test-audio-i')\
                ([text_emb_i, audio_emb_i])
        audio_image_imag = Lambda(
            lambda x: 0.5*x[0]+0.5*x[1]+K.sqrt(x[0]*x[1]+1e-7),
            name='audio-image-i')\
                ([audio_emb_i, image_emb_i])

        text_image_real =\
            Lambda(lambda x: K.expand_dims(x, axis=1))(text_image_real)
        text_audio_real =\
            Lambda(lambda x: K.expand_dims(x, axis=1))(text_audio_real)
        audio_image_real =\
            Lambda(lambda x: K.expand_dims(x, axis=1))(audio_image_real)
        text_image_imag =\
            Lambda(lambda x: K.expand_dims(x, axis=1))(text_image_imag)
        text_audio_imag =\
            Lambda(lambda x: K.expand_dims(x, axis=1))(text_audio_imag)
        audio_image_imag =\
            Lambda(lambda x: K.expand_dims(x, axis=1))(audio_image_imag)

        # audio_emb_r =\
        #     Lambda(lambda x: K.expand_dims(x, axis=1))(audio_emb_r)
        # image_emb_r =\
        #     Lambda(lambda x: K.expand_dims(x, axis=1))(image_emb_r)
        # text_emb_r =\
        #     Lambda(lambda x: K.expand_dims(x, axis=1))(text_emb_r)
        # audio_emb_i =\
        #     Lambda(lambda x: K.expand_dims(x, axis=1))(audio_emb_i)
        # image_emb_i =\
        #     Lambda(lambda x: K.expand_dims(x, axis=1))(image_emb_i)
        # text_emb_i =\
        #     Lambda(lambda x: K.expand_dims(x, axis=1))(text_emb_i)

        real = Concatenate(axis=1)(
            [text_image_real, text_audio_real, audio_image_real])
        imag = Concatenate(axis=1)(
            [text_image_imag, text_audio_imag, audio_image_imag])
        # real = Concatenate(axis=1)(
        #     [audio_emb_r, image_emb_r, text_emb_r])
        # imag = Concatenate(axis=1)(
        #     [audio_emb_i, image_emb_i, text_emb_i])
        print('real.shape 0:', real.shape)
        print('image.shape 0:', imag.shape)
        real =\
            Lambda(lambda x: K.mean(x, axis=1))(real)
        imag =\
            Lambda(lambda x: K.mean(x, axis=1))(imag)
        print('real.shape 1:', real.shape)
        print('image.shape 1:', imag.shape)
        # print('text_emb_r.shape:', text_emb_r.shape)
        # print('text_emb_i.shape:', text_emb_i.shape)
        # print('image_emb_r.shape:', image_emb_r.shape)
        # print('image_emb_i.shape:', image_emb_i.shape)
        # print('audio_emb_r.shape:', audio_emb_r.shape)
        # print('audio_emb_i.shape:', audio_emb_i.shape)

        probs_task1 = self.projection_task1(
            [real, imag])
        probs_task2 = self.projection_task2(
            [real, imag])
        probs_task3 = self.projection_task3(
            [real, imag])

        probs_task1 = Dense(
            512,
            activation='relu',
            kernel_initializer='glorot_normal')(probs_task1)
        probs_task1 = self.dropout_probs(probs_task1)
        probs_task1 = Dense(
            128,
            activation='relu',
            kernel_initializer='glorot_normal')(probs_task1)
        probs_task1 = self.dropout_probs(probs_task1)

        probs_task2 = Dense(
            512,
            activation='relu',
            kernel_initializer='glorot_normal')(probs_task2)
        probs_task2 = self.dropout_probs(probs_task2)
        probs_task2 = Dense(
            128,
            activation='relu',
            kernel_initializer='glorot_normal')(probs_task2)
        probs_task2 = self.dropout_probs(probs_task2)

        probs_task3 = Dense(
            512,
            activation='relu',
            kernel_initializer='glorot_normal')(probs_task3)
        probs_task3 = self.dropout_probs(probs_task3)
        probs_task3 = Dense(
            128,
            activation='relu',
            kernel_initializer='glorot_normal')(probs_task3)
        probs_task3 = self.dropout_probs(probs_task3)
        return probs_task1, probs_task2, probs_task3
