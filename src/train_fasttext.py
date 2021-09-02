import numpy
import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd
import torch
import configparser
from all_parameters_sentiment import get_all_parameters
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from workspace_cls_kw import SENT_WORDID, SENT_LABELID, SENT_WORD_MASK, SENT_ORIGINAL_TXT, KWS_IDS, KWS_IDF
from torch.utils.data import Dataset, DataLoader, RandomSampler, SubsetRandomSampler
import argparse
from utils_torch_cls_kw import compute_values, get_data, compute_values_eval
from experiment_imax_kw_sentiment import RunExperiment
from workspace_cls_kw import workspace
from model_imax_kw_sentiment import *
from vocabulary_cls import get_word_info
import math
import random

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


import nltk
nltk.data.path.append("~/nltk_data/")
import re
import string

from gensim.models.word2vec import Word2Vec, LineSentence
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors

import fasttext
import fasttext.util
from gensim.models.fasttext import load_facebook_model
from gensim.models.fasttext import FastText as FT_gensim
from gensim.test.utils import datapath
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from random import shuffle

from wikipedia2vec import Wikipedia2Vec

from pprint import pprint
from copy import deepcopy

import time
from datetime import datetime, timedelta
from gensim import utils, matutils

# NLTK Stop words
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import _pickle as cPickle

def read_pickle(filepath, filename):

        f = open(os.path.join(filepath, filename), 'rb')
        read_file = cPickle.load(f)
        f.close()

        return read_file

def save_pickle(filepath, filename, data):

    f = open(os.path.join(filepath, filename), 'wb')
    cPickle.dump(data, f)
    print(" file saved to: %s"%(os.path.join(filepath, filename)))
    f.close()



def get_stop_words(stop_file_path):
    """load stop words """
    
    with open(stop_file_path, 'r', encoding="utf-8") as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
        return frozenset(stop_set)


def cleaning_text(txt):

    punct = ''.join([p for p in string.punctuation])

    txt = txt.replace('i.e.', 'id est')
    txt = txt.replace('e.g.', 'exempli gratia')
    txt = txt.lower().replace('q&a', 'question and answer')
    txt = txt.replace('&', 'and')
    txt = re.sub(r'@\w+', '', txt)
    txt = re.sub(r'[-+]?[.\d]*[\d]+[:,.\d]*', '', txt)
    txt = re.sub(r'[^\x00-\x7f]', '', txt)
    txt = re.sub(r'\b\w{1}\b', '', txt)
    txt = re.sub(r'\b\w{20.1000}\b', '', txt)
    regex = re.compile('[%s]' % re.escape(punct)) 
    txt = regex.sub(' ', txt)
    txt = ' '.join(txt.split())

    return txt

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:
        fname = feature_names[idx]
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results



def get_amazondata():

    xdat = []
    PATH = '~/data/AmazonDat/'

    print("Processing Apps_for_Android ...")
    sys.stdout.flush()
    with open(PATH+'train/Apps_for_Android.train') as f:
        for line in f:
            xdat.append(line.split('\t')[0])

    print("Processing Beauty ...")
    sys.stdout.flush()
    with open(PATH+'train/Beauty.train') as f:
        for line in f:
            xdat.append(line.split('\t')[0])

    print("Processing Books ...")
    sys.stdout.flush()
    with open(PATH+'train/Books.train') as f:
        for line in f:
            xdat.append(line.split('\t')[0])

    print("Processing CDs_and_Vinyl ...")
    sys.stdout.flush()
    with open(PATH+'train/CDs_and_Vinyl.train') as f:
        for line in f:
            xdat.append(line.split('\t')[0])

    print("Processing Cell_Phones_and_Accessories ...")
    sys.stdout.flush()
    with open(PATH+'train/Cell_Phones_and_Accessories.train') as f:
        for line in f:
            xdat.append(line.split('\t')[0])

    print("Processing Clothing_Shoes_and_Jewelry ...")
    sys.stdout.flush()
    with open(PATH+'train/Clothing_Shoes_and_Jewelry.train') as f:
        for line in f:
            xdat.append(line.split('\t')[0])

    print("Processing Electronics ...")
    sys.stdout.flush()
    with open(PATH+'train/Electronics.train') as f:
        for line in f:
            xdat.append(line.split('\t')[0])

    print("Processing Health_and_Personal_Care ...")
    sys.stdout.flush()
    with open(PATH+'train/Health_and_Personal_Care.train') as f:
        for line in f:
            xdat.append(line.split('\t')[0])

    print("Processing Home_and_Kitchen ...")
    sys.stdout.flush()
    with open(PATH+'train/Home_and_Kitchen.train') as f:
        for line in f:
            xdat.append(line.split('\t')[0])
            
    print("Processing Kindle_Store ...")
    sys.stdout.flush()
    with open(PATH+'train/Kindle_Store.train') as f:
        for line in f:
            xdat.append(line.split('\t')[0])

    print("Processing Movies_and_TV ...")
    sys.stdout.flush()
    with open(PATH+'train/Movies_and_TV.train') as f:
        for line in f:
            xdat.append(line.split('\t')[0])

    print("Processing Office_Products ...")
    sys.stdout.flush()
    with open(PATH+'train/Office_Products.train') as f:
        for line in f:
            xdat.append(line.split('\t')[0])

    print("Processing Sports_and_Outdoors ...")
    sys.stdout.flush()
    with open(PATH+'train/Sports_and_Outdoors.train') as f:
        for line in f:
            xdat.append(line.split('\t')[0])

    clean_xdat = []
    for txt in xdat:
        txt = cleaning_text(txt)
        clean_xdat.append(txt)


    return clean_xdat




def get_intentdata():

    xdat = []
    PATH = '~/data/IntentDat/'

    print("Processing Assistant ...")
    sys.stdout.flush()

    with open(PATH+'train/Assistant.train') as f:
        for line in f:
            xdat.append(line.split('\t')[0])

    print("Processing Atis ...")
    sys.stdout.flush()
    with open(PATH+'train/Atis.train') as f:
        for line in f:
            xdat.append(line.split('\t')[0])

    print("Processing Finance ...")
    sys.stdout.flush()
    with open(PATH+'train/Finance.train') as f:
        for line in f:
            xdat.append(line.split('\t')[0])

    print("Processing Stackoverflow ...")
    sys.stdout.flush()
    with open(PATH+'train/Stackoverflow.train') as f:
        for line in f:
            xdat.append(line.split('\t')[0])


    clean_xdat = []
    for txt in xdat:
        txt = cleaning_text(txt)
        clean_xdat.append(txt)


    return clean_xdat

def finetuning_amazon(data):

    print("Training Word Embeddings on Amazon data set...")


    model = load_facebook_model('~/embeddings/cc.en.100.bin')

    oldmodel = deepcopy(model)

    data = [t.split() for t in data]

    n_sents = len(data)
   
    model.build_vocab(data, update=True)
    model.train(data, total_examples=n_sents, epochs=20)
    model.save('~/embeddings/w2v_fasttext_sentiment.model')

    for m in ['oldmodel', 'model']:
        print('The vocabulary size of the w2v_fasttext_cls', m, 'is', len(eval(m).wv.vocab))
        sys.stdout.flush()


def finetuning_intent(data):

    print("Training Word Embeddings on Intent data set...")


    model = load_facebook_model('~/embeddings/cc.en.100.bin')

    oldmodel = deepcopy(model)

    data = [t.split() for t in data]

    n_sents = len(data)
   
    model.build_vocab(data, update=True)
    model.train(data, total_examples=n_sents, epochs=20)
    model.save('~/embeddings/w2v_fasttext_intent.model')

    for m in ['oldmodel', 'model']:
        print('The vocabulary size of the w2v_fasttext_cls', m, 'is', len(eval(m).wv.vocab))
        sys.stdout.flush()


# scripts for loading pretrained word embedding model
def load_w2v_sentiment(PATH):

    vocab_new = []
    word_vecs_new = []
    zeros_init = [float(0.)] * 100

    model = Word2Vec.load('~/embeddings/w2v_fasttext_sentiment.model')
    vocab = list(model.wv.vocab)
    word_vecs = model.wv.vectors
    w2v = model.wv


    if '</s>' in vocab:
        idx = vocab.index('</s>')
        vocab[idx] = '</s2>'

    if '<unk>' in vocab:
        idx = vocab.index('<unk>')
        vocab[idx] = '<unk2>'

    word_vecs = word_vecs.tolist()
    
    vocab_new.append('</s>') #0
    word_vecs_new.append(zeros_init)
    vocab_new.append('<unk>') #1
    word_vecs_new.append(zeros_init)

    vocab_new.extend(vocab)
    word_vecs_new.extend(word_vecs)

    word_vecs_new = np.array(word_vecs_new)

    return vocab_new, word_vecs_new


def load_w2v_intent(PATH):

    vocab_new = []
    word_vecs_new = []
    zeros_init = [float(0.)] * 100

    model = Word2Vec.load('~/embeddings/w2v_fasttext_intent.model')
    vocab = list(model.wv.vocab)
    word_vecs = model.wv.vectors
    w2v = model.wv


    if '</s>' in vocab:
        idx = vocab.index('</s>')
        vocab[idx] = '</s2>'

    if '<unk>' in vocab:
        idx = vocab.index('<unk>')
        vocab[idx] = '<unk2>'

    word_vecs = word_vecs.tolist()
    
    vocab_new.append('</s>') #0
    word_vecs_new.append(zeros_init)
    vocab_new.append('<unk>') #1
    word_vecs_new.append(zeros_init)

    vocab_new.extend(vocab)
    word_vecs_new.extend(word_vecs)

    word_vecs_new = np.array(word_vecs_new)

    return vocab_new, word_vecs_new

def get_per_domain(PATH, domain):

    #Apps_for_Android.train

    xdat = []
    ydat = []

    print("Processing domain ...", domain)
    sys.stdout.flush()
    with open(PATH+domain) as f:
        for line in f:
            xdat.append(line.split('\t')[0])
            ydat.append(line.split('\t')[1][0])

    clean_xdat = []
    for txt in xdat:
        txt = cleaning_text(txt)
        clean_xdat.append(txt)

    return clean_xdat, ydat

def extract_kws(PATH, domain, params):

    cv = params['cv']
    tfidf_transformer = params['tfidf_transformer']
    word2idx = params['vocabulary']
    UNKNOWN_WORD_INDEX = word2idx['<unk>']
    PAD = word2idx['</s>']

    text, y = get_per_domain(PATH, domain)

    print("Extracting keywords ...")
    sys.stdout.flush()

    feature_names=cv.get_feature_names()
    tf_idf_vector=tfidf_transformer.transform(cv.transform(text))

    results_keywords_id =[]
    results_vals =[]
    results_kws =[]

    for i in range(tf_idf_vector.shape[0]):

        # get vector for a single document
        curr_vector=tf_idf_vector[i]
        
        #sort the tf-idf vector by descending order of scores
        sorted_items=sort_coo(curr_vector.tocoo())

        #extract only the top n; n here is 10
        keywords=extract_topn_from_vector(feature_names,sorted_items,10) 

        kws = list(keywords.keys())
        vals = list(keywords.values())

        keywords_id = []
        for kw, val in keywords.items():
            if kw in word2idx:
                keywords_id.append(word2idx[kw])
            else:
                keywords_id.append(word2idx['<unk>'])

        if len(keywords_id) < 10:
            len_ = 10 - len(keywords_id)
            for _ in range(len_):
                keywords_id.append(word2idx['</s>'])
                vals.append(0.)

        results_keywords_id.append(keywords_id)
        results_vals.append(vals)
        results_kws.append(kws)
    
    df = pd.DataFrame(zip(text, y, results_keywords_id, results_vals, results_kws), columns=['doc','label', 'keywords_id', 'vals', 'kws'])

    path = PATH+domain

    base=os.path.basename(path)
    fn = os.path.splitext(base)[0]

    np.savetxt('~/data/Amazondat/train/Kws_%s.train'%fn, df.values, fmt='%s', delimiter='\t')
Amazondat/



if __name__ == '__main__':


    parser = argparse.ArgumentParser(description="Extracting Keywords for ProtoInfoMax++ from sentiment dataset ...")
    parser.add_argument('-config', help="path to configuration file", 
                        default="./config")
    parser.add_argument('-section', help="the section name of the experiment")

    args = parser.parse_args()
    config_paths = [args.config]
    config_parser = configparser.SafeConfigParser()
    config_found = config_parser.read(config_paths)

    params = get_all_parameters(config_parser, args.section)
    params['model_string'] = args.section

    numpy.random.seed(params['seed'])
    random.seed(params['seed'])

    print('Parameters:', params)
    sys.stdout.flush()

    data = get_amazondata()
    
    # Continue training word2vec on sentiment data
    finetuning_amazon(data)

    voc, w2v = load_w2v_sentiment()

    idx2word = dict([(i, voc[i]) for i in range(len(voc))])
    word2idx = dict([(v,k) for k,v in idx2word.items()])
    save_pickle('~/data/dict_idx2word_sentiment.pkl', (word2idx, idx2word))


    data_intent = get_intentdata()
    
    # Continue training word2vec on sentiment data
    finetuning_intent(data_intent)

    voc, w2v = load_w2v_intent()

    idx2word = dict([(i, voc[i]) for i in range(len(voc))])
    word2idx = dict([(v,k) for k,v in idx2word.items()])
    save_pickle('~/data/dict_idx2word_intent.pkl', (word2idx, idx2word))

      