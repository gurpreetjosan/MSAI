#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 06:01:30 2018

@author: gurpreet
"""
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, Bidirectional
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Embedding,Masking
from keras.layers.merge import concatenate
# define the captioning model

#this is model that combine img feature withs caption feature and pass it to LSTM layer. We have 2 LSTM layers here

def define_model_BILSTM2L(vocab_size, max_query_length, max_para_length,num_classes):
    modelname="BiLSTM2L-Dense-Model_"
    # query embedding

    inputs2 = Input(shape=(max_query_length,))
    mask = Masking(mask_value=0)(inputs2)
    emb2 = Embedding(vocab_size, 200, mask_zero=True)(mask)
    emb3 = Bidirectional(LSTM(units=250, return_sequences=True, recurrent_dropout=0.1))(emb2)
    emb3 = Bidirectional(LSTM(units=250, recurrent_dropout=0.1))(emb3)
    emb3 = Dropout(0.5)(emb3)
    #emb4 = TimeDistributed(Dense(128, activation='relu'))(emb3)

    #para embedding

    #src_txt_length = max_length_news

    inputs3 = Input(shape=(max_para_length,))
    mask = Masking(mask_value=0)(inputs3)
    encoder1 = Embedding(vocab_size, 200,mask_zero=True)(mask)
    encoder2 = Bidirectional(LSTM(units=250, return_sequences=True, recurrent_dropout=0.1))(encoder1)
    encoder2 = Bidirectional(LSTM(units=250, recurrent_dropout=0.1))(encoder2)
    encoder2=Dropout(0.5)(encoder2)
    #encoder3 = RepeatVector(sum_txt_length)(encoder2)

    # merge inputs
    merged = concatenate([emb3, encoder2])
    merged=Dropout(0.5)(merged)
    # language model (decoder)
    lm2 = merged #LSTM(250)(merged)
    lm3 = Dense(250, activation='relu')(lm2)
    outputs = Dense(num_classes, activation='softmax')(lm3)
    # tie it together [image, seq, news] [word]
    model = Model(inputs=[inputs2, inputs3], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #print(model.summary())
    plot_model(model, to_file='./graphs/'+modelname+'.png', show_shapes=True)
    return model,modelname

def define_model_BILSTM1L(vocab_size, max_query_length, max_para_length,num_classes):
    modelname="BiLSTM1L-Dense-Model_"

    # query embedding
    inputs2 = Input(shape=(max_query_length,))
    mask = Masking(mask_value=0)(inputs2)
    emb2 = Embedding(vocab_size, 200, mask_zero=True)(mask)
    #emb3 = Bidirectional(LSTM(units=250, return_sequences=True, recurrent_dropout=0.1))(emb2)
    emb3 = Bidirectional(LSTM(units=250, recurrent_dropout=0.1))(emb2)
    emb3 = Dropout(0.5)(emb3)
    #emb4 = TimeDistributed(Dense(128, activation='relu'))(emb3)

    #para embedding
    inputs3 = Input(shape=(max_para_length,))
    mask = Masking(mask_value=0)(inputs3)
    encoder1 = Embedding(vocab_size, 200,mask_zero=True)(mask)
    #encoder2 = Bidirectional(LSTM(units=250, return_sequences=True, recurrent_dropout=0.1))(encoder1)
    encoder2 = Bidirectional(LSTM(units=250, recurrent_dropout=0.1))(encoder1)
    encoder2=Dropout(0.5)(encoder2)
    #encoder3 = RepeatVector(sum_txt_length)(encoder2)

    # merge inputs
    merged = concatenate([emb3, encoder2])
    merged=Dropout(0.5)(merged)
    # Dense Neural Network
    dnn = Dense(250, activation='relu')(merged)
    outputs = Dense(num_classes, activation='softmax')(dnn)
    # tie it together [query, text] [label]
    model = Model(inputs=[inputs2, inputs3], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #print(model.summary())
    plot_model(model, to_file='./graphs/'+modelname+'.png', show_shapes=True)
    return model,modelname

