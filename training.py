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

queryTrain, queryV, paraTrain, paraV, labelTrain, labelV = util.loadData_new()
print(len(queryTrain),len(paraTrain),len(labelTrain))
print(len(queryV),len(paraV),len(labelV))

queryTrain = {key:queryTrain[key] for i, key in enumerate(queryTrain) if i <5000}
print(len(queryTrain))
queryV={key:queryV[key] for i, key in enumerate(queryV) if i <500}
print("Tokenizing...")


if os.path.isfile("./data/tokenizer.pickle"):
    with open('./data/tokenizer.pickle', 'rb') as handle:
        tokenizer=pickle.load(handle)
else:
    tokenizer=util.create_tokenizer([queryTrain,queryV],[paraTrain,paraV])

vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)


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
n_epochs = 15
n_queries_per_update = 7
n_batches_per_epoch = int(len(queryTrain) / n_queries_per_update)
n_repeats = 1
validationSteps = int(len(queryV) / n_queries_per_update)
resume_trng= 'no'
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
    print("if part")

else:
    # define the model
    model,modelFileName = models.define_model_BILSTM1L(vocab_size, max_query_length, max_para_length,num_classes)
    for i in range(n_repeats):
        # define checkpoint callback
        filepath = 'trained_model/'+modelFileName+'best.h5'#-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1)
        earlystop=EarlyStopping(monitor="val_loss",patience=2)
        # fit model
        history =model.fit_generator(
            util.data_generator(queryTrain,paraTrain,labelTrain, tokenizer, max_query_length, max_para_length,
                           n_queries_per_update),
            validation_data= util.data_generator(queryV, paraV, labelV, tokenizer, max_query_length,
                           max_para_length, n_queries_per_update),
            validation_steps=validationSteps,
            steps_per_epoch=n_batches_per_epoch,
            epochs=n_epochs,
            verbose=verbose,
            callbacks=[checkpoint,earlystop])

        hist = pd.DataFrame(history.history)
        dump(hist, open(modelFileName + '-history.pkl', 'wb'))

        plt.style.use("ggplot")
        fig = plt.figure(figsize=(12, 12))
        # plt.plot(hist["loss"])
        plt.plot(hist["acc"])
        # plt.plot(hist["val_loss"])
        plt.plot(hist["val_acc"])
        plt.show()
        # plt.ylabel('Frequency [Hz]')
        # plt.xlabel('Time [sec]')
        plt.savefig('./graphs/' + modelFileName + "plt")
        fig.savefig('./graphs/' + modelFileName + "fig")
        plt.savefig(modelFileName + "-graph.png", bbox_inches='tight')
        fig.savefig(modelFileName + "-graph.png", bbox_inches='tight')
        #model.save('model_' + str(i) + '.h5')
        # evaluate model on training data
        #train_score = evaluate_model(model, train_descriptions, train_features, train_newstext, tokenizer, max_length,max_length_news)
        #test_score = evaluate_model(model, test_descriptions, test_features, test_newstext, tokenizer, max_length,max_length_news)
        # store
        #train_results.append(train_score)
        #test_results.append(test_score)
        #print('>%d: train=%f test=%f' % ((i + 1), train_score, test_score))
        # save results to fil
        #df = DataFrame()
        #df['train'] = train_results
        #df['test'] = test_results
        #print(df.describe())
        #df.to_csv(model_name + '.csv', index=False)
    

print("done")
