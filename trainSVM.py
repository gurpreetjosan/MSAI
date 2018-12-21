#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 18:28:48 2018

@author: gurpreet
"""
import util
import models
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint,EarlyStopping
from pickle import load, dump
import os.path, pickle

queryTrain, queryV, paraTrain, paraV, labelTrain, labelV = util.loadStemedTrainDataFromPickle()
print(len(queryTrain),len(paraTrain),len(labelTrain))
print(len(queryV),len(paraV),len(labelV))


#logic for reducing training samples and evaluation samples

queryTrain = {key:queryTrain[key] for i, key in enumerate(queryTrain) if (i>0 and i<250000)}
paraTrain = {key:paraTrain[key] for i, key in enumerate(queryTrain)}

print(len(queryTrain),len(paraTrain))
queryV={key:queryV[key] for i, key in enumerate(queryV) if i <20000}
paraV={key:paraV[key] for i, key in enumerate(queryV)}
#----------------------------------------------
filterSample="2L"

print("Tokenizing...")


if os.path.isfile("./data/tokenizer-"+filterSample+"queries.pickle"):
    #tokenizer = util.create_tokenizer([queryTrain, queryV], [paraTrain, paraV],"1L")
    with open('./data/tokenizer-'+filterSample+'queries.pickle', 'rb') as handle:
        tokenizer=pickle.load(handle)
else:
    print("No tokenizer dump exist. creating tokenizer file..")# from dividefile.py and restart this program.")
    #exit(0)
    tokenizer=util.create_tokenizer([queryTrain,queryV],[paraTrain,paraV],"2L")

vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)

emb_dim=200
embeddings_index =util.load_embedding(emb_dim)
embedding_matrix=util.create_weight_matrix(vocab_size,tokenizer,embeddings_index,emb_dim)

del embeddings_index #free memory
# determine the maximum train sequence length from both train and validate data
print("Finding max para length...")
max_para_length=362

# -----------------------------------------------------------------------------
#use below logic if don't know maximum para length
'''
max_para_length=0
for ls in paraTrain.values():
    for s in ls:
        if len(s.split())>max_para_length:
            max_para_length=len(s.split())
for ls in paraV.values():
    for s in ls:
        if len(s.split())>max_para_length:
            max_para_length=len(s.split())

#max_para_length = max([len(s.split()) for s in lst] for lst in paras.values())
'''
print('Maximum passage Length: %d' % max_para_length)

# -----------------------------------------------------------------------------

max_query_length=38
#use below line if don't know maximum query length
#max_query_length = max(max(len(s.split()) for s in list(queryTrain.values())), max(len(s.split()) for s in list(queryV.values())))
print('Maximum query Length: %d' % max_query_length)
# -----------------------------------------------------------------------------
# define experiment
verbose = 1
n_epochs = 5
n_queries_per_update = 10
n_batches_per_epoch = int(len(queryTrain) / n_queries_per_update)
n_repeats = 1
validationSteps = int(len(queryV) / n_queries_per_update)
resume_trng= 'yes'
steps = len(queryTrain)
num_classes=2
# -----------------------------------------------------------------------------
# check working of generator
#generator = util.data_generator(queryTrain,paraTrain,labelTrain, tokenizer, max_query_length,max_para_length,1)
#inputs, outputs = next(generator)
#print(inputs[0].shape)
#print(inputs[1].shape)
#print(outputs.shape)
# -----------------------------------------------------------------------------

#run experiment

print("starting Training....Finger Crossed....")
if resume_trng =='yes':
    from keras.models import load_model
    model = load_model('./trained_model/BiLSTM1L-Dense-Model_Vcab_913344Emb_200best.h5')

else:
    # define the model
    model,modelFileName = models.define_model_BILSTM1L_withAttentionSVM(vocab_size, max_query_length, max_para_length,num_classes,embedding_matrix,emb_dim)
    modelFileName = modelFileName + "Vcab_"+str(vocab_size) + "Emb_"+ str(emb_dim)

from keras import backend as K

# with a Sequential model
get_dense_layer_output = K.function([model.layers[0].input], [model.layers[11].output])

layer_output = get_dense_layer_output([x])[0]