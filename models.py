#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 06:01:30 2018

@author: gurpreet
"""
from keras.backend import expand_dims
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, Bidirectional, Permute, Reshape, Lambda, merge, K, multiply
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Embedding,Masking
from keras.layers.merge import concatenate
from keras_contrib.layers import CRF
# define the captioning model

# if True, the attention vector is shared across the input_dimensions where the attention is applied.
SINGLE_ATTENTION_VECTOR = False
APPLY_ATTENTION_BEFORE_LSTM = False

def attention_3d_block(inputs,TIME_STEPS):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1))(a)
    output_attention_mul = multiply([inputs, a_probs])
    return output_attention_mul

#this is model that combine img feature withs caption feature and pass it to LSTM layer. We have 2 LSTM layers here

def define_model_BILSTM2L(vocab_size, max_query_length, max_para_length,num_classes,emb_dim):
    modelname="BiLSTM2L-Dense-Model_"
    # query embedding

    inputs2 = Input(shape=(max_query_length,))
    mask = Masking(mask_value=0)(inputs2)
    emb2 = Embedding(vocab_size, emb_dim, mask_zero=True)(mask)
    emb3 = Bidirectional(LSTM(units=250, return_sequences=True, recurrent_dropout=0.1))(emb2)
    emb3 = Bidirectional(LSTM(units=250, recurrent_dropout=0.1))(emb3)
    emb3 = Dropout(0.5)(emb3)
    #emb4 = TimeDistributed(Dense(128, activation='relu'))(emb3)

    #para embedding

    #src_txt_length = max_length_news

    inputs3 = Input(shape=(max_para_length,))
    mask = Masking(mask_value=0)(inputs3)
    encoder1 = Embedding(vocab_size, emb_dim,mask_zero=True)(mask)
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

def define_model_BILSTM1L(vocab_size, max_query_length, max_para_length,num_classes,embedding_matrix,emb_dim):
    modelname="BiLSTM1L-Dense-Model_"

    # query embedding
    inputs2 = Input(shape=(max_query_length,))
    mask = Masking(mask_value=0)(inputs2)
    emb2 = Embedding(vocab_size, emb_dim, weights=[embedding_matrix], trainable=False, mask_zero=True)(mask)
    #emb3 = Bidirectional(LSTM(units=250, return_sequences=True, recurrent_dropout=0.1))(emb2)
    emb3 = Bidirectional(LSTM(units=250, recurrent_dropout=0.1))(emb2)
    emb3 = Dropout(0.5)(emb3)
    #emb4 = TimeDistributed(Dense(128, activation='relu'))(emb3)

    #para embedding
    inputs3 = Input(shape=(max_para_length,))
    mask = Masking(mask_value=0)(inputs3)
    encoder1 = Embedding(vocab_size, emb_dim, weights=[embedding_matrix], trainable=False,mask_zero=True)(mask)
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

def define_model_BILSTM1L_withAttentionSVM(vocab_size, max_query_length, max_para_length,num_classes,embedding_matrix,emb_dim):
    modelname="BiLSTM1L-Dense-Model-withAtntn+SVM_"

    # query embedding
    inputs2 = Input(shape=(max_query_length,))
    #mask = Masking(mask_value=0)(inputs2)
    emb2 = Embedding(vocab_size, emb_dim, weights=[embedding_matrix], trainable=False)(inputs2)
    attention_mul = attention_3d_block(emb2,max_query_length)
    #emb3 = Bidirectional(LSTM(units=250, return_sequences=True, recurrent_dropout=0.1))(emb2)
    emb3 = Bidirectional(LSTM(units=250, recurrent_dropout=0.1))(attention_mul)
    emb3 = Dropout(0.3)(emb3)
    #emb4 = TimeDistributed(Dense(128, activation='relu'))(emb3)

    #para embedding
    inputs3 = Input(shape=(max_para_length,))
    #mask = Masking(mask_value=0)(inputs3)
    encoder1 = Embedding(vocab_size, emb_dim, weights=[embedding_matrix], trainable=False)(inputs3)
    attention_mul = attention_3d_block(encoder1, max_para_length)
    #encoder2 = Bidirectional(LSTM(units=250, return_sequences=True, recurrent_dropout=0.1))(encoder1)
    encoder2 = Bidirectional(LSTM(units=250, recurrent_dropout=0.1))(attention_mul)
    encoder2=Dropout(0.3)(encoder2)
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
    #plot_model(model, to_file='./graphs/'+modelname+'.png', show_shapes=True)
    return model,modelname


def define_model_BILSTM1L_withSVM(vocab_size, max_query_length, max_para_length,num_classes,embedding_matrix,emb_dim):
    modelname="BiLSTM1L-Dense-Model-withSVM_"

    # query embedding
    inputs2 = Input(shape=(max_query_length,))
    mask = Masking(mask_value=0)(inputs2)
    emb2 = Embedding(vocab_size, emb_dim, weights=[embedding_matrix], trainable=False, mask_zero=True)(mask)
    attention_mul = attention_3d_block(emb2,max_query_length)
    #emb3 = Bidirectional(LSTM(units=250, return_sequences=True, recurrent_dropout=0.1))(emb2)
    emb3 = Bidirectional(LSTM(units=250, recurrent_dropout=0.1))(attention_mul)
    emb3 = Dropout(0.5)(emb3)
    #emb4 = TimeDistributed(Dense(128, activation='relu'))(emb3)

    #para embedding
    inputs3 = Input(shape=(max_para_length,))
    mask = Masking(mask_value=0)(inputs3)
    encoder1 = Embedding(vocab_size, emb_dim, weights=[embedding_matrix], trainable=False,mask_zero=True)(mask)
    attention_mul = attention_3d_block(encoder1, max_para_length)
    #encoder2 = Bidirectional(LSTM(units=250, return_sequences=True, recurrent_dropout=0.1))(encoder1)
    encoder2 = Bidirectional(LSTM(units=250, recurrent_dropout=0.1))(attention_mul)
    encoder2=Dropout(0.5)(encoder2)
    #encoder3 = RepeatVector(sum_txt_length)(encoder2)

    # merge inputs
    merged = concatenate([emb3, encoder2])
    merged=Dropout(0.5)(merged)
    from sklearn.svm import SVC
    svc=SVC(kernel='linear')(merged)#SVC(kernel='linear')(merged)
    # Dense Neural Network

    #dnn = Dense(250, activation='relu')(merged)
    #outputs = Dense(num_classes, activation='softmax')(dnn)
    # tie it together [query, text] [label]
    model = Model(inputs=[inputs2, inputs3], outputs=svc)
    model.compile(loss='hinge', optimizer='adam', metrics=['accuracy'])
    #print(model.summary())
    plot_model(model, to_file='./graphs/'+modelname+'.png', show_shapes=True)
    return model,modelname


def define_model_BILSTM1L_withCRF(vocab_size, max_query_length, max_para_length,num_classes,embedding_matrix,emb_dim):
    modelname="BiLSTM1L-Dense-Model-withCRF_"

    # query embedding
    inputs2 = Input(shape=(max_query_length,))
    mask = Masking(mask_value=0)(inputs2)
    emb2 = Embedding(vocab_size, emb_dim, weights=[embedding_matrix], trainable=False, mask_zero=True)(mask)

    #emb3 = Bidirectional(LSTM(units=250, return_sequences=True, recurrent_dropout=0.1))(emb2)
    emb3 = Bidirectional(LSTM(units=250, recurrent_dropout=0.1,return_sequences=True))(emb2)
    emb3 = Dropout(0.3)(emb3)
    #emb4 = TimeDistributed(Dense(128, activation='relu'))(emb3)

    #para embedding
    inputs3 = Input(shape=(max_para_length,))
    mask = Masking(mask_value=0)(inputs3)
    encoder1 = Embedding(vocab_size, emb_dim, weights=[embedding_matrix], trainable=False,mask_zero=True)(mask)
    #encoder2 = Bidirectional(LSTM(units=250, return_sequences=True, recurrent_dropout=0.1))(encoder1)
    encoder2 = Bidirectional(LSTM(units=250, recurrent_dropout=0.1,return_sequences=True))(encoder1 )
    encoder2=Dropout(0.3)(encoder2)
    #encoder3 = RepeatVector(sum_txt_length)(encoder2)

    # merge inputs
    merged = concatenate([emb3, encoder2],axis=1)
    #merged=Dropout(0.5)(merged)
    # Dense Neural Network
    #dnn = Dense(250, activation='relu')(merged)

    crf = CRF(2)  # CRF layer
    #merged=expand_dims(merged, axis=2)

    outputs = crf(merged)  # output
    #outputs = Dense(num_classes, activation='softmax')(dnn)
    # tie it together [query, text] [label]
    model = Model(inputs=[inputs2, inputs3], outputs=outputs)
    model.compile(optimizer="adam", loss=crf.loss_function, metrics=[crf.accuracy])
    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #print(model.summary())
    plot_model(model, to_file='./graphs/'+modelname+'.png', show_shapes=True)
    return model,modelname
