import sys

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

def loadData_new():
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


def loadData(datafolder):
    # global GloveEmbeddings,emb_dim,max_query_words,max_passage_words
    query_id_list=set()
    barlength=100
    queryTrain_mapping,queryV_mapping = dict(),dict()
    labelTrain_mapping,labelV_mapping = dict(),dict()
    paraTrain_mapping,paraV_mapping = dict(),dict()
    data_folder = Path(datafolder)
    #inputfile = data_folder
    #make list of all query ids
    print("creating query id list")
    f = open(str(data_folder), "r", encoding="utf-8", errors="ignore")  # Format of the file : query_id \t query \t passage \t label \t passage_id
    for line in f:
        tokens = line.strip().lower().split("\t")
        if tokens[0] not in query_id_list:
            query_id_list.add(tokens[0])
    f.close()
    query_id_list = list(query_id_list)
    shuffle(query_id_list)
    train=query_id_list[:int(.8 * len(query_id_list))]
    validate=query_id_list[int(.8 * len(query_id_list)):]
    print("query id list prepared. Total queries: " + str(len(query_id_list)))

    query_count=0
    f = open(str(data_folder), "r", encoding="utf-8",errors="ignore")  # Format of the file : query_id \t query \t passage \t label \t passage_id
    #    fw = open(outputfile,"w",encoding="utf-8")
    for line in f:
        tokens = line.strip().lower().split("\t")
        query_id, query, passage, label = tokens[0], tokens[1], tokens[2], tokens[3]
        # ****Query Processing****
        #words = re.split('\W+', query)
        #words = [x for x in words if x]  # to remove empty words
        #words = clean_text(words)

        if query_id in train:
            if query_id not in queryTrain_mapping:
                queryTrain_mapping[query_id] = query#words
                #print("Reading query " + query_id)
                query_count=query_count+1
        elif query_id in validate:
            if query_id not in queryV_mapping:
                queryV_mapping[query_id] = query#words
                #print("Reading query " + query_id)
                query_count = query_count + 1

        # ****Para Processing****
        #words = re.split('\W+', passage)
        #words = [x for x in words if x]  # to remove empty words
        #words = clean_text(words)
        # Add para to dictionary
        if query_id in train:
            if query_id not in paraTrain_mapping:
                paraTrain_mapping[query_id] = list()
            paraTrain_mapping[query_id].append(passage)#(words)
        elif query_id in validate:
            if query_id not in paraV_mapping:
                paraV_mapping[query_id] = list()
            paraV_mapping[query_id].append(passage)#(words)

        # Add labels to dictionary
        if query_id in train:
            if query_id not in labelTrain_mapping:
                labelTrain_mapping[query_id] = list()
            labelTrain_mapping[query_id].append(label)
        if query_id in validate:
            if query_id not in labelV_mapping:
                labelV_mapping[query_id] = list()
            labelV_mapping[query_id].append(label)

        text = "\r{0} {1}".format("loaded queries: ",query_count)
        sys.stdout.write(text)
        sys.stdout.flush()
        #print("%age completed " + str(perc))
    return queryTrain_mapping,queryV_mapping,paraTrain_mapping,paraV_mapping,labelTrain_mapping,labelV_mapping


def clean_text(desc):
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    # desc = desc.split()
    # convert to lower case
    desc = [word.lower() for word in desc]
    # remove punctuation from each token
    desc = [w.translate(table) for w in desc]
    # remove hanging 's' and 'a'
    desc = [word for word in desc if len(word) > 1]
    # remove tokens with numbers in them
    #desc = [word for word in desc if word.isalpha()]
    # store as string
    return ' '.join(desc)

# fit a tokenizer given query and para descriptions. Used if tokens from corpus is generated

def create_tokenizer(querylist,datalist):

    tokenizer = Tokenizer(oov_token="<OOV>")
    print("adding queries")
    for d in querylist:
        tokenizer.fit_on_texts([" ".join(d.values())])

    print("adding paras")
    for d in datalist:
        " ".join(" ".join(x) for x in d.values())
        #for sl in d:
        #    tokenizer.fit_on_texts([" ".join(sl.values())])

    with open('./data/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return tokenizer

# load the whole embedding into memory
def load_embedding():
    embeddings_index = dict()
    f = open('glove.6B.100d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = numpy.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index


# create a weight matrix for words in training docs
def create_weight_matrix(vocab_size, t, ei):  # t is pointing to tokenizer, ei is embedding_index
    embedding_matrix = numpy.zeros((vocab_size, 100))
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
                print(query)
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
