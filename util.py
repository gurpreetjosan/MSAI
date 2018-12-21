import sys
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas
import pickle
from keras.preprocessing.text import Tokenizer
import numpy
import os
import string
from pathlib import Path
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from numpy import array
from random import shuffle

from pandas import DataFrame

def loadEvalData():
    print("loading queriesE from pickle files")
    with open('./data/queryE.pickle', 'rb') as handle:
        queryTrain_mapping = pickle.load(handle)
    print("loading paraT from pickle files")
    with open('./data/paraE.pickle', 'rb') as handle:
        paraTrain_mapping = pickle.load(handle)
    return  queryTrain_mapping, paraTrain_mapping

def loadTrainDataFromPickle():
    if any(File.endswith(".pickle") for File in os.listdir("./data/")):
        print("loading queriesT from pickle files")
        with open('./data/queryTrain.pickle', 'rb') as handle:
            queryTrain_mapping = pickle.load(handle)
        print("loading paraT from pickle files")
        with open('./data/paraTrain.pickle', 'rb') as handle:
            paraTrain_mapping=pickle.load(handle)
        print("loading labelT from pickle files")
        with open('./data/labelTrain.pickle', 'rb') as handle:
            labelTrain_mapping=pickle.load(handle)
        print("loading queriesV from pickle files")
        with open('./data/queryV.pickle', 'rb') as handle:
            queryV_mapping = pickle.load(handle)
        print("loading paraV from pickle files")
        with open('./data/paraV.pickle', 'rb') as handle:
            paraV_mapping=pickle.load(handle)
        print("loading labelV from pickle files")
        with open('./data/labelV.pickle', 'rb') as handle:
            labelV_mapping=pickle.load(handle)

    else:
        import dividefile
        print ("pickle files not exists. creating it....")
        dividefile.pickleData()
        print ("pickle files created. Rerun program...")
        exit()

    return queryTrain_mapping,queryV_mapping,paraTrain_mapping,paraV_mapping,labelTrain_mapping,labelV_mapping

def loadStemedTrainDataFromPickle():
    print("loading queriesT from pickle files")
    with open('./data/queryTrain_stemed.pickle', 'rb') as handle:
        queryTrain_mapping = pickle.load(handle)
    print("loading paraT from pickle files")
    with open('./data/paraTrain_stemed.pickle', 'rb') as handle:
        paraTrain_mapping=pickle.load(handle)
    print("loading labelT from pickle files")
    with open('./data/labelTrain_stemed.pickle', 'rb') as handle:
       labelTrain_mapping=pickle.load(handle)
    print("loading queriesV from pickle files")
    with open('./data/queryV_stemed.pickle', 'rb') as handle:
       queryV_mapping = pickle.load(handle)
    print("loading paraV from pickle files")
    with open('./data/paraV_stemed.pickle', 'rb') as handle:
       paraV_mapping=pickle.load(handle)
    print("loading labelV from pickle files")
    with open('./data/labelV_stemed.pickle', 'rb') as handle:
       labelV_mapping=pickle.load(handle)

    return queryTrain_mapping,queryV_mapping,paraTrain_mapping,paraV_mapping,labelTrain_mapping,labelV_mapping

def clean_text(desc):
    # prepare translation table for removing punctuation
    #table = str.maketrans('', '', string.punctuation)
    # desc = desc.split()
    # convert to lower case
    #desc = [word.lower() for word in desc]
    # remove punctuation from each token
    #desc = [w.translate(table) for w in desc]
    # remove hanging 's' and 'a'
    desc = [word for word in desc if len(word) > 1]
    # remove tokens with numbers in them
    #desc = [word for word in desc if word.isalpha()]
    # store as string
    return ' '.join(desc)

# fit a tokenizer given query and para descriptions. Used if tokens from corpus is generated
def create_tokenizer(querylist,datalist,size): #size is number of queries for which tokenizer is working
    tokenizer = Tokenizer(oov_token="<OOV>")
    print("fitting queries....")
    for d in querylist:
        tokenizer.fit_on_texts([" ".join(d.values())])
    print("fitting paras....")
    for d in datalist:
        temp=[x for v in d.values() for x in v]
        tokenizer.fit_on_texts(temp)

    with open('./data/tokenizer-'+size+'queries.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return tokenizer

# load the whole embedding into memory
def load_embedding(emb_dim):
    print("loading Glove index...")
    embeddings_index = dict()
    f = open('./Glove/glove.6B.'+str(emb_dim)+'d.txt',encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = numpy.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index

# create a weight matrix for words in training docs
def create_weight_matrix(vocab_size, t, ei,emb_dim):  # t is pointing to tokenizer, ei is embedding_index
    embedding_matrix = numpy.random.random((vocab_size, emb_dim))
    for word, i in t.word_index.items():
        embedding_vector = ei.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def data_generator(queries, paras, labels, tokenizer, max_query_length, max_para_length, n_step):

    # loop until we finish training
    while 1:
        # loop over query identifiers in the dataset
        keys = list(queries.keys())
        for i in range(0, len(keys), n_step):
            Xquery, Xpara, y = list(), list(), list()
            for j in range(i, min(len(keys), i + n_step)):
                query_id = keys[j]
                # retrieve query
                query = queries[query_id]
                #print(query)
                query=clean_text([query])
                # retrieve text input
                paralist = paras[query_id]
                cleanparalist=list()
                for k in range(len(paralist)):
                    cleanparalist.append(clean_text([paralist[k]]))
                # generate input-output pairs
                query_seq = tokenizer.texts_to_sequences([query])[0]
                para_seq = tokenizer.texts_to_sequences(cleanparalist)
                # split one sequence into multiple X,y pairs
                padded_query_seq = pad_sequences([query_seq], maxlen=max_query_length)[0]
                padded_para_seq = pad_sequences(para_seq, maxlen=max_para_length)
                out_seq = labels[query_id]

                for ff in range(len(paralist)):
                    Xquery.append(padded_query_seq)
                    Xpara.append(padded_para_seq[ff])
                    y.append(to_categorical(out_seq[ff], num_classes=2))
            # yield this batch of samples to the model
            yield [[array(Xquery), array(Xpara)], array(y)]
