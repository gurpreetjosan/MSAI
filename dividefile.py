import datetime
import re
import string
import sys
from pathlib import Path
from random import shuffle
import pickle,pandas

from keras_preprocessing.text import Tokenizer
from nltk import SnowballStemmer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

def divideFile():
    query_id_list = set()
    linecnt=0
    data_folder="./data/data.tsv"
    data_folder = Path(data_folder)
    # inputfile = data_folder
    # make list of all query ids
    print("creating query id list")
    f = open(str(data_folder), "r", encoding="utf-8",errors="ignore")  # Format of the file : query_id \t query \t passage \t label \t passage_id
    for line in f:
        tokens = line.strip().lower().split("\t")
        linecnt=linecnt+1
        if tokens[0] not in query_id_list:
            query_id_list.add(tokens[0])
    f.close()
    print("total lines: "+ str(linecnt))

    query_id_list = list(query_id_list)
    shuffle(query_id_list)
    train = query_id_list[:int(.8 * len(query_id_list))]
    validate = query_id_list[int(.8 * len(query_id_list)):]
    fw_t = open("./data/traindata.tsv", "w", encoding="utf-8")
    fw_v = open("./data/validationdata.tsv", "w", encoding="utf-8")
    print("creating seperate files")

    f = open(str(data_folder), "r", encoding="utf-8",errors="ignore")  # Format of the file : query_id \t query \t passage \t label \t passage_id
    linecnt = 0
    print(datetime.datetime.now().time())

    for line in f:
        line = re.sub(r'([@#$(){}_]*)([0-9]+)([#$%(),.:?_]+)([0-9]*)([@#${}().?!%_]*)', r' \2\4 ',line)  # remove punctuation from numerals
        line = re.sub(r'([a-zA-Z])([!"#$%&()*+,-./:;<=>?@[\]^_`{|}~])', r'\1 ', line)  # remove punct from frontside
        line = re.sub(r'([!"#$%&()*+,-./:;<=>?@[\]^_`{|}~])([a-zA-Z])', r' \2', line)  # remove punct from backside
        line = re.sub(r'( )([!"#$%&()*+,-./:;<=>?@[\]^_`{|}~])+', r'', line)  # remove dangling punct
        line = re.sub(r'([!"#$%&()*+,-./:;<=>?@[\]^_`{|}~])+( )', r'', line)  # remove dangling punct
        line = line.replace("'", " ")
        line = re.sub(' {2,}', ' ', line)
        tokens = line.strip().lower().split("\t")
        linecnt = linecnt + 1
        if tokens[0] in validate:
            fw_v.write(line )
        else:
            fw_t.write(line)

        if linecnt%5000==0:
            text = "\r{0} {1} {2}".format("loaded lines: ", linecnt,datetime.datetime.now().time())
            sys.stdout.write(text)
            sys.stdout.flush()
            fw_v.flush()
            fw_t.flush()


    f.close()
    fw_v.close()
    fw_t.close()
    print("done")


def pickleData():
    labelTrain_mapping, labelV_mapping,labelE_mapping = dict(), dict(), dict()
    paraTrain_mapping, paraV_mapping,paraE_mapping = dict(), dict(), dict()

    #stop = stopwords.words('english')# if want to remove stop words
    # Loading train data
    print("Loading Train data...")
    df = pandas.read_csv('./data/traindata.tsv', sep='\t',header=None)

    #remove stop words
    #df[1] = df[1].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    # df[2] = df[2].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

    queryTrain_mapping = df.set_index(0)[1].to_dict()
    print("Loading ParaTrain data...")
    for x in range(len(df)):
        currentid = df.iloc[x, 0]
        currentvalue = df.iloc[x, 2]
        paraTrain_mapping.setdefault(currentid, [])
        paraTrain_mapping[currentid].append(currentvalue)
    print("Loading labelTrain data...")
    for x in range(len(df)):
        currentid = df.iloc[x, 0]
        currentvalue = df.iloc[x, 3]
        labelTrain_mapping.setdefault(currentid, [])
        labelTrain_mapping[currentid].append(currentvalue)

    print("Pickling Train data...")
    with open('./data/queryTrain.pickle', 'wb') as handle:
        pickle.dump(queryTrain_mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('./data/paraTrain.pickle', 'wb') as handle:
        pickle.dump(paraTrain_mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('./data/labelTrain.pickle', 'wb') as handle:
        pickle.dump(labelTrain_mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #cleaning memory
    del queryTrain_mapping
    del paraTrain_mapping
    del labelTrain_mapping

    print("Loading validation data...")

    # Loading validation data
    df = pandas.read_csv('./data/validationdata.tsv', sep='\t', header=None)
    # remove stop words
    # df[1] = df[1].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    # df[2] = df[2].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

    queryV_mapping = df.set_index(0)[1].to_dict()
    print("Loading paraV data...")
    for x in range(len(df)):
        currentid = df.iloc[x, 0]
        currentvalue = df.iloc[x, 2]
        paraV_mapping.setdefault(currentid, [])
        paraV_mapping[currentid].append(currentvalue)
    print("Loading labelV data...")
    for x in range(len(df)):
        currentid = df.iloc[x, 0]
        currentvalue = df.iloc[x, 3]
        labelV_mapping.setdefault(currentid, [])
        labelV_mapping[currentid].append(currentvalue)

    print("Pickling validation data...")
    with open('./data/queryV.pickle', 'wb') as handle:
        pickle.dump(queryV_mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('./data/paraV.pickle', 'wb') as handle:
        pickle.dump(paraV_mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('./data/labelV.pickle', 'wb') as handle:
        pickle.dump(labelV_mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # Loading evaluation data
    df = pandas.read_csv('eval1_unlabelled.tsv', sep='\t', header=None)
    # remove stop words
    # df[1] = df[1].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    # df[2] = df[2].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

    queryE_mapping = df.set_index(0)[1].to_dict()
    print("Loading paraE data...")
    for x in range(len(df)):
        currentid = df.iloc[x, 0]
        currentvalue = df.iloc[x, 2]
        paraE_mapping.setdefault(currentid, [])
        paraE_mapping[currentid].append(currentvalue)

    print("Pickling evaluation data...")
    with open('./data/queryE.pickle', 'wb') as handle:
        pickle.dump(queryE_mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('./data/paraE.pickle', 'wb') as handle:
        pickle.dump(paraE_mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)

def pickletokenizer():
    tokenizer = Tokenizer(oov_token="<OOV>")
    print("Loading Train data...")
    df = pandas.read_csv('./data/traindata.tsv', sep='\t', header=None)
    p = df[2].tolist()
    print("adding paras")
    tokenizer.fit_on_texts(p)

    print("adding queries")
    p = df[1].tolist()
    tokenizer.fit_on_texts(p)

    print("Loading Validation data...")
    df = pandas.read_csv('./data/validationdata.tsv', sep='\t', header=None)
    p = df[2].tolist()
    print("adding paras")
    tokenizer.fit_on_texts(p)

    print("adding queries")
    p = df[1].tolist()
    tokenizer.fit_on_texts(p)

    with open('./data/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_StemedData():
    labelTrain_mapping, labelV_mapping,labelE_mapping = dict(), dict(), dict()
    paraTrain_mapping, paraV_mapping,paraE_mapping = dict(), dict(), dict()
    table = str.maketrans('', '', string.punctuation)
    ps = nltk.PorterStemmer()
    stop = []#stopwords.words('english')

    # Loading train data
    print("Loading Train data...")
    df = pandas.read_csv('./data/traindata.tsv', sep='\t',header=None)
    #apply stemming on the whole query and para column
    print("applying stemming on query...")
    df[1] = df[1].apply(lambda x: " ".join([ps.stem(word.lower().translate(table)) for word in x.split(" ") if word not in stop]))

    print("applying stemming on paras...")
    df[2] = df[2].apply(lambda x: " ".join([ps.stem(word.lower().translate(table)) for word in x.split(" ") if word not in stop]))

    queryTrain_mapping = df.set_index(0)[1].to_dict()
    print("Loading ParaTrain data...")
    for x in range(len(df)):
        currentid = df.iloc[x, 0]
        currentvalue = df.iloc[x, 2]
        paraTrain_mapping.setdefault(currentid, [])
        paraTrain_mapping[currentid].append(currentvalue)
    print("Loading labelTrain data...")
    for x in range(len(df)):
        currentid = df.iloc[x, 0]
        currentvalue = df.iloc[x, 3]
        labelTrain_mapping.setdefault(currentid, [])
        labelTrain_mapping[currentid].append(currentvalue)

    print("Pickling Train data...")
    with open('./data/queryTrain_stemed.pickle', 'wb') as handle:
        pickle.dump(queryTrain_mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('./data/paraTrain_stemed.pickle', 'wb') as handle:
        pickle.dump(paraTrain_mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('./data/labelTrain_stemed.pickle', 'wb') as handle:
        pickle.dump(labelTrain_mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #cleaning memory
    del queryTrain_mapping
    del paraTrain_mapping
    del labelTrain_mapping

    print("Loading validation data...")

    # Loading validation data
    df = pandas.read_csv('./data/validationdata.tsv', sep='\t', header=None)
    print("applying stemming on query...")
    df[1] = df[1].apply(
        lambda x: " ".join([ps.stem(word.lower().translate(table)) for word in x.split(" ") if word not in stop]))
    print("applying stemming on para...")
    df[2] = df[2].apply(
        lambda x: " ".join([ps.stem(word.lower().translate(table)) for word in x.split(" ") if word not in stop]))

    queryV_mapping = df.set_index(0)[1].to_dict()
    print("Loading paraV data...")
    for x in range(len(df)):
        currentid = df.iloc[x, 0]
        currentvalue = df.iloc[x, 2]
        paraV_mapping.setdefault(currentid, [])
        paraV_mapping[currentid].append(currentvalue)
    print("Loading labelV data...")
    for x in range(len(df)):
        currentid = df.iloc[x, 0]
        currentvalue = df.iloc[x, 3]
        labelV_mapping.setdefault(currentid, [])
        labelV_mapping[currentid].append(currentvalue)

    print("Pickling validation data...")
    with open('./data/queryV_stemed.pickle', 'wb') as handle:
        pickle.dump(queryV_mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('./data/paraV_stemed.pickle', 'wb') as handle:
        pickle.dump(paraV_mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('./data/labelV_stemed.pickle', 'wb') as handle:
        pickle.dump(labelV_mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)

    del queryV_mapping
    del paraV_mapping
    del labelV_mapping

    # Loading evaluation data
    df = pandas.read_csv('eval1_unlabelled.tsv', sep='\t', header=None)
    print("applying stemming on query...")
    df[1] = df[1].apply(
        lambda x: " ".join([ps.stem(word.lower().translate(table)) for word in x.split(" ") if word not in stop]))
    print("applying stemming on para...")
    df[2] = df[2].apply(
        lambda x: " ".join([ps.stem(word.lower().translate(table)) for word in x.split(" ") if word not in stop]))

    queryE_mapping = df.set_index(0)[1].to_dict()
    print("Loading paraE data...")
    for x in range(len(df)):
        currentid = df.iloc[x, 0]
        currentvalue = df.iloc[x, 2]
        paraE_mapping.setdefault(currentid, [])
        paraE_mapping[currentid].append(currentvalue)

    print("Pickling evaluation data...")
    with open('./data/queryE_stemed.pickle', 'wb') as handle:
        pickle.dump(queryE_mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('./data/paraE_stemed.pickle', 'wb') as handle:
        pickle.dump(paraE_mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)

def pickle_NostopwrdData():
    labelTrain_mapping, labelV_mapping,labelE_mapping = dict(), dict(), dict()
    paraTrain_mapping, paraV_mapping,paraE_mapping = dict(), dict(), dict()
    table = str.maketrans('', '', string.punctuation)
    #ps = SnowballStemmer("english")
    stop = stopwords.words('english')

    # Loading train data
    print("Loading Train data...")
    df = pandas.read_csv('./data/traindata.tsv', sep='\t',header=None)
    #apply stemming on the whole query and para column
    print("applying stemming on query...")
    df[1] = df[1].apply(lambda x: " ".join([word.lower().translate(table) for word in x.split(" ") if word not in stop]))
    print("applying stemming on paras...")
    df[2] = df[2].apply(lambda x: " ".join([word.lower().translate(table) for word in x.split(" ") if word not in stop]))

    queryTrain_mapping = df.set_index(0)[1].to_dict()
    print("Loading ParaTrain data...")
    for x in range(len(df)):
        currentid = df.iloc[x, 0]
        currentvalue = df.iloc[x, 2]
        paraTrain_mapping.setdefault(currentid, [])
        paraTrain_mapping[currentid].append(currentvalue)
    print("Loading labelTrain data...")
    for x in range(len(df)):
        currentid = df.iloc[x, 0]
        currentvalue = df.iloc[x, 3]
        labelTrain_mapping.setdefault(currentid, [])
        labelTrain_mapping[currentid].append(currentvalue)

    print("Pickling Train data...")
    with open('./data/queryTrain_Nostopwrd.pickle', 'wb') as handle:
        pickle.dump(queryTrain_mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('./data/paraTrain_Nostopwrd.pickle', 'wb') as handle:
        pickle.dump(paraTrain_mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('./data/labelTrain_Nostopwrd.pickle', 'wb') as handle:
        pickle.dump(labelTrain_mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #cleaning memory
    del queryTrain_mapping
    del paraTrain_mapping
    del labelTrain_mapping

    print("Loading validation data...")

    # Loading validation data
    df = pandas.read_csv('./data/validationdata.tsv', sep='\t', header=None)
    print("applying stemming on query...")
    df[1] = df[1].apply(
        lambda x: " ".join([word.lower().translate(table) for word in x.split(" ") if word not in stop]))
    print("applying stemming on para...")
    df[2] = df[2].apply(
        lambda x: " ".join([word.lower().translate(table) for word in x.split(" ") if word not in stop]))

    queryV_mapping = df.set_index(0)[1].to_dict()
    print("Loading paraV data...")
    for x in range(len(df)):
        currentid = df.iloc[x, 0]
        currentvalue = df.iloc[x, 2]
        paraV_mapping.setdefault(currentid, [])
        paraV_mapping[currentid].append(currentvalue)
    print("Loading labelV data...")
    for x in range(len(df)):
        currentid = df.iloc[x, 0]
        currentvalue = df.iloc[x, 3]
        labelV_mapping.setdefault(currentid, [])
        labelV_mapping[currentid].append(currentvalue)

    print("Pickling validation data...")
    with open('./data/queryV_Nostopwrd.pickle', 'wb') as handle:
        pickle.dump(queryV_mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('./data/paraV_Nostopwrd.pickle', 'wb') as handle:
        pickle.dump(paraV_mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('./data/labelV_Nostopwrd.pickle', 'wb') as handle:
        pickle.dump(labelV_mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)

    del queryV_mapping
    del paraV_mapping
    del labelV_mapping

    # Loading evaluation data
    df = pandas.read_csv('eval1_unlabelled.tsv', sep='\t', header=None)
    print("applying stemming on query...")
    df[1] = df[1].apply(
        lambda x: " ".join([word.lower().translate(table) for word in x.split(" ") if word not in stop]))
    print("applying stemming on para...")
    df[2] = df[2].apply(
        lambda x: " ".join([word.lower().translate(table) for word in x.split(" ") if word not in stop]))

    queryE_mapping = df.set_index(0)[1].to_dict()
    print("Loading paraE data...")
    for x in range(len(df)):
        currentid = df.iloc[x, 0]
        currentvalue = df.iloc[x, 2]
        paraE_mapping.setdefault(currentid, [])
        paraE_mapping[currentid].append(currentvalue)

    print("Pickling evaluation data...")
    with open('./data/queryE_Nostopwrd.pickle', 'wb') as handle:
        pickle.dump(queryE_mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('./data/paraE_Nostopwrd.pickle', 'wb') as handle:
        pickle.dump(paraE_mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)

#divideFile()
pickle_StemedData()