import numpy
import sys
import re
from keras import preprocessing
import pickle
from numpy import array
from keras_preprocessing.sequence import pad_sequences
from keras.models import load_model

import util
def GetPredictionOnEvalSet(modelfilename,tokenizer):
    global q_max_words,p_max_words,emb_dim
    max_para_length = 362
    max_query_length = 38
    model = load_model('./trained_model/'+modelfilename+'.h5')
    #f = open(testfile,'r',encoding="utf-8")
    all_scores={} # Dictionary with key = query_id and value = array of scores for respective passages
    queryE, paraE= util.loadEvalData()
    if tokenizer==None:
        with open('./data/tokenizer-Stemed2Lqueries.pickle', 'rb') as handle:
            tokenizer=pickle.load(handle)
    keys = list(queryE.keys())
    print(len(keys))
    for i in range(len(keys)):
        Xquery, Xpara = list(), list()
        query_id = keys[i]
        # retrieve query
        query = queryE[query_id]
        query = util.clean_text([query])
        # retrieve text input
        paralist = paraE[query_id]
        cleanparalist = list()
        for j in range(len(paralist)):
            cleanparalist.append(util.clean_text([paralist[j]]))
        # generate input-output pairs
        query_seq = tokenizer.texts_to_sequences([query])[0]
        para_seq = tokenizer.texts_to_sequences(cleanparalist)
        # split one sequence into multiple X,y pairs
        padded_query_seq = pad_sequences([query_seq], maxlen=max_query_length)[0]
        #print(padded_query_seq)
        padded_para_seq = pad_sequences(para_seq, maxlen=max_para_length)
        for k in range(len(paralist)):
            Xquery.append(padded_query_seq)
            Xpara.append(padded_para_seq[k])

        score = model.predict([array(Xquery), array(Xpara)],verbose=0) # do forward-prop on model to get score
        score=score[:,1] # extract 1 column at index 1
        if(query_id in all_scores):
            all_scores[query_id].append(score)
        else:
            all_scores[query_id] = [score]
        text = "\r{0} {1}".format("Done queries: ", i)
        sys.stdout.write(text)
        sys.stdout.flush()


    fw = open("answer.tsv","w",encoding="utf-8")
    for query_id in all_scores:
        scores = all_scores[query_id]
        s=""
        for sc in scores:
            for value in sc:
                value=format(value,'f')#.replace("\n","").replace("[","").replace("]","")
                s=s+ str(value) + "\t"
        s=re.sub(' {2,}', ' ', s)
        s=re.sub(' ', '\t', s)
        fw.write(str(query_id) + "\t" +s.rstrip("\t") +  "\n")
        '''
        scores_str = [str(sc) for sc in scores] # convert all scores to string values
        
        scores_str = "\t".join(scores_str) # join all scores in list to make it one string with  tab delimiter.
        re.sub("[|]", "", scores_str)
        scores_str.replace(" ","\t")
        fw.write(str(query_id)+"\t"+ scores_str + "\n")
        '''
    fw.close()
GetPredictionOnEvalSet('BiLSTM1L-Dense-Model-withAtntn_Vcab_913344Emb_200best',None)