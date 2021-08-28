import numpy
import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import torch

import configparser
from all_parameters_intent import get_all_parameters
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from workspace_intent_kw import SENT_WORDID, SENT_LABELID, SENT_WORD_MASK, SENT_ORIGINAL_TXT, KWS_IDS, KWS_IDF
from torch.utils.data import Dataset, DataLoader, RandomSampler, SubsetRandomSampler
import argparse
from utils_torch_intent_kw import compute_values, get_data, compute_values_eval
from experiment_imax_kw_intent import RunExperiment
from workspace_intent_kw import workspace
from model_imax_kw_intent import *
from vocabulary_intent import get_word_info
import math
import random

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import nltk
nltk.data.path.append(NLTK_PATH)
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



def load_w2v(PATH):

    vocab_new = []
    word_vecs_new = []
    zeros_init = [float(0.)] * 100

    model = Word2Vec.load(PATH+'w2v_fasttext_intent.model')
    vocab = list(model.wv.vocab)
    word_vecs = model.wv.vectors
    w2v = model.wv

    print("len vocab:", len(vocab))
    sys.stdout.flush()
    print("vocab", vocab[:50])
    sys.stdout.flush()
    print("word_vecs shape:", word_vecs.shape)
    sys.stdout.flush()

    word_vecs = word_vecs.tolist()
    
    vocab_new.append('</s>') #0
    word_vecs_new.append(zeros_init)
    vocab_new.append('<unk>') #1
    word_vecs_new.append(zeros_init)

    vocab_new.extend(vocab)
    word_vecs_new.extend(word_vecs)

    word_vecs_new = np.array(word_vecs_new)

    print("len vocab after OOV:", len(vocab_new))
    print("word_vecs shape after OOV:", word_vecs_new.shape)

    return vocab_new, word_vecs_new


class TestLoader(Dataset):
    def __init__(self, data, params):

        self.data = data
        
        self.params = params
        self.batch_size = self.params['batch_size']
        self.X_sup = None
        self.X_sup_mask = None
        self.Y_sup = None
        self.X_q = None
        self.X_q_mask = None
        self.Y_q = None
        self.Y_sup_onehot  = None
        self.Y_q_onehot = None

        self.pair_data = self._build_pairs()

    def get_one_hot(self, y_target, classe_list):

        cls_id_map = dict()
        for lid in classe_list:
                    cls_id_map[lid] = len(cls_id_map)


        y_target_one_hot = numpy.zeros([len(y_target), len(cls_id_map)])
        for k in range(len(y_target)):
            y_target_one_hot[k][cls_id_map[y_target[k]]] = 1.0

        return y_target_one_hot

    def get_supporting_set(self, test_workspace):

        all_ss_info = test_workspace.get_flatten_supporting_set()[:self.params['min_ss_size']]

        strbatch = [s[SENT_ORIGINAL_TXT] for s in
                    all_ss_info]
        txtbatch = [s[SENT_WORDID] for s in
                    all_ss_info]
        maskbatch = [s[SENT_WORD_MASK] for s in
                     all_ss_info]
        ybatch = [s[SENT_LABELID] for s in
                  all_ss_info]
       
        nClasses = 2
        ybatch_one_shot = self.get_one_hot(ybatch, range(nClasses))

        
        return strbatch, txtbatch, maskbatch, ybatch, ybatch_one_shot

    def _build_pairs(self):


        all_examples = []
       
        domain_examples = []
       
        for b in range(len(self.data)):
            # for sampling query (get n-query per intent/class)
            qcurr_dev_workspace = self.data[b]
            nClasses = 2

            for target_set, target_set_file in zip(qcurr_dev_workspace.target_sets, qcurr_dev_workspace.target_sets_files):                  
                #print('Testing: ', target_set_file)
                if str(target_set_file).find('test') != -1:
                                        
                    input_target_sent_test= [target_sentence[SENT_WORDID] for target_sentence in target_set]
                    in_txt= [target_sentence[SENT_ORIGINAL_TXT] for target_sentence in target_set]
                    target_sent_mask_test= [target_sentence[SENT_WORD_MASK] for target_sentence in target_set]
                    groundtruth_id = [target_sentence[SENT_LABELID] for target_sentence in target_set]
                    groundtruth = [qcurr_dev_workspace.lblist[gt_id] for gt_id in groundtruth_id]


                    input_target_sent_test = np.array(input_target_sent_test)
                    in_txt = np.array(in_txt)
                    target_sent_mask_test = np.array(target_sent_mask_test)
                    groundtruth_id = np.array(groundtruth_id)
                    groundtruth = np.array(groundtruth)
                   

                   
                    for idx in range(len(input_target_sent_test)):

                        X_q = numpy.array(input_target_sent_test[idx], dtype='int')
                        X_q_mask = numpy.array(target_sent_mask_test[idx], dtype='int')
                        Y_q = numpy.array(groundtruth_id[idx], dtype='int')

                        len_ = np.count_nonzero(X_q)
                        self.len_X_q = [len_]
                        self.len_X_q = np.array(self.len_X_q)

                    
                        self.X_q = numpy.array([X_q])
                        self.X_q_mask = numpy.array([X_q_mask])
                        self.Y_q = numpy.array([Y_q])


                        # sampling support examples
                        self.X_sup = []
                        self.X_sup_mask = []
                        self.Y_sup_onehot = []
                        self.Y_sup =[]
                        self.len_X_sup = []

                        # for sampling support
                        
                        nClasses = 2
                        sup_str, support_txtid, support_mask, support_ybatch, support_ybatch_one_shot = self.get_supporting_set(qcurr_dev_workspace)

                        support_txtid= np.array(support_txtid)
                        support_ybatch_one_shot= np.array(support_ybatch_one_shot)
                        support_ybatch= np.array(support_ybatch)
                        support_mask= np.array(support_mask)
                     
                        
                        support_txtid = support_txtid.reshape((1,self.params['min_ss_size'],-1))
                        support_ybatch_one_shot = support_ybatch_one_shot.reshape((1,self.params['min_ss_size'],-1))
                        support_ybatch = support_ybatch.reshape((1,self.params['min_ss_size']))
                        support_mask = support_mask.reshape((1,self.params['min_ss_size'], -1))
                        

                        len_X_sup = []
                        for sup in support_txtid:
                            lens = []
                            for d in sup:
                                lens.append(np.count_nonzero(d))
                            len_X_sup.append(lens)

                        self.X_sup = numpy.array(support_txtid.tolist())
                        self.X_sup_mask = numpy.array(support_mask.tolist())
                        self.Y_sup_onehot = numpy.array(support_ybatch_one_shot.tolist())
                        self.Y_sup = numpy.array(support_ybatch.tolist())
                        self.len_X_sup = numpy.array(len_X_sup)
                      

                        example_dict = {'X_sup': torch.from_numpy(self.X_sup.astype(np.int64)), 'X_sup_len': torch.from_numpy(self.len_X_sup), 'Y_sup': torch.from_numpy(self.Y_sup), 'Y_sup_oh': torch.from_numpy(self.Y_sup_onehot.astype(np.int64)), 'X_q':torch.from_numpy(self.X_q.astype(np.int64)), 'X_q_len': torch.from_numpy(self.len_X_q), 'Y_q': torch.from_numpy(self.Y_q), 'target_set_file': target_set_file}

                        domain_examples.append(example_dict)

            all_examples.append(domain_examples)

        return all_examples

    def __len__(self):
        return len(self.pair_data)

    def __getitem__(self, index):
        return self.pair_data[index]

class AmazonLoader(Dataset):
    def __init__(self, data, params):

        self.data = data
        
        self.params = params
        self.batch_size = self.params['batch_size']
        self.X_sup = None
        self.X_sup_mask = None
        self.Y_sup = None
        self.X_q = None
        self.X_q_mask = None
        self.Y_q = None
        self.X_neg = None
        self.X_neg_mask = None
        self.selected_labels = None
        self.Y_sup_onehot  = None
        self.Y_q_onehot = None

        self.pair_data = self._build_pairs()

    def get_train_batch(self, train_workspace, all_ood_workspaces):

        sent_list = train_workspace.target_sets[0]
        nClasses_in_Train = len(train_workspace.labels_in_train)

        sampled_class_size = int(self.params['sampling_classes'])
        sampled_classes = train_workspace.sample_classes(sampled_class_size)

        sentence_size_per_intent = max(1, int(self.params['min_ss_size']))
        cls_ = np.arange(10)

        sent_id_batch = []
        for b in range(100):
            selected_label = random.choice(sampled_classes)

            selected_utt = random.choice(train_workspace.train_intent2ids_list[selected_label])
            sent_id_batch.append(int(selected_utt))


        x_target_wid = [sent_list[i][SENT_WORDID] for i in sent_id_batch]
        y_target = [sent_list[i][SENT_LABELID] for i in sent_id_batch]
        x_target_mask = [sent_list[i][SENT_WORD_MASK] for i in sent_id_batch]

        x_target_kws = [sent_list[i][KWS_IDS] for i in sent_id_batch]
        x_target_kwsidf = [sent_list[i][KWS_IDF] for i in sent_id_batch]

        x_support_set_wid = []
        y_support_set = []
        x_support_set_mask = []
        x_support_kws = []
        x_support_kwsidf = []
        x_ood_wid = []
        x_ood_mask = []
        y_ood = []

        x_ood_kws = []
        x_ood_kwsidf= []

        for target_sid in range(len(sent_id_batch)):
            selected_ood_sent_infos = []
            for _ in range(self.params['ood_example_size']):
                selected_ood_workspace = numpy.random.choice(all_ood_workspaces)
                fss = selected_ood_workspace.get_flatten_supporting_set()
                selected_id = numpy.random.choice([i for i in range(len(fss))])
                selected_ood_sent_info = fss[selected_id]
                selected_ood_sent_infos.append(selected_ood_sent_info)

            ss_sent_info = train_workspace.select_support_set(sentence_size_per_intent,
                                                              0,
                                                              sent_id_batch[target_sid], sampled_classes)


            x_support_set_wid_per_sent = [sinfo[SENT_WORDID] for sinfo in ss_sent_info]
            y_support_set_per_sent = [sinfo[SENT_LABELID] for sinfo in ss_sent_info]
            x_support_set_mask_per_sent = [sinfo[SENT_WORD_MASK] for sinfo in ss_sent_info]

            x_support_kws_per_sent = [sinfo[KWS_IDS] for sinfo in ss_sent_info]
            x_support_kwsidf_per_sent = [sinfo[KWS_IDF] for sinfo in ss_sent_info]


            x_support_set_wid_per_sent= np.array(x_support_set_wid_per_sent)
            y_support_set_per_sent= np.array(y_support_set_per_sent)
            x_support_set_mask_per_sent= np.array(x_support_set_mask_per_sent)

            x_support_kws_per_sent = np.array(x_support_kws_per_sent)
            x_support_kwsidf_per_sent = np.array(x_support_kwsidf_per_sent)
            
            x_support_set_wid_per_sent = x_support_set_wid_per_sent.reshape((10,self.params['min_ss_size'],-1))
            y_support_set_per_sent = y_support_set_per_sent.reshape((10,self.params['min_ss_size']))
            x_support_set_mask_per_sent = x_support_set_mask_per_sent.reshape((10,self.params['min_ss_size'], -1))
            x_support_kws_per_sent = x_support_kws_per_sent.reshape((10,self.params['min_ss_size'],-1))
            x_support_kwsidf_per_sent = x_support_kwsidf_per_sent.reshape((10,self.params['min_ss_size'],-1))

            x_support_set_mask.extend(x_support_set_mask_per_sent.tolist())
            x_support_set_wid.extend(x_support_set_wid_per_sent.tolist())
            y_support_set.extend(y_support_set_per_sent.tolist())
            x_support_kws.extend(x_support_kws_per_sent.tolist())
            x_support_kwsidf.extend(x_support_kwsidf_per_sent.tolist())

            x_ood_wid_per_sent = [sinfo[SENT_WORDID] for sinfo in selected_ood_sent_infos]
            x_ood_mask_per_sent = [sinfo[SENT_WORD_MASK] for sinfo in selected_ood_sent_infos]
            x_ood_label_per_sent = [sinfo[SENT_LABELID] for sinfo in selected_ood_sent_infos]

            x_ood_kws_per_sent = [sinfo[KWS_IDS] for sinfo in selected_ood_sent_infos]
            x_ood_kwsidf_per_sent = [sinfo[KWS_IDF] for sinfo in selected_ood_sent_infos]

            x_ood_wid.append(x_ood_wid_per_sent)
            x_ood_mask.append(x_ood_mask_per_sent)
            y_ood.append(x_ood_label_per_sent)

            x_ood_kws.append(x_ood_kws_per_sent)
            x_ood_kwsidf.append(x_ood_kwsidf_per_sent)

        y_ood = np.array(y_ood)
        y_ood = y_ood.reshape((len(sent_id_batch))).tolist()
        
        return x_support_set_wid, \
            x_support_kws, \
            x_support_kwsidf, \
            y_support_set,  \
            x_support_set_mask,  \
            x_target_wid, \
            x_target_kws, \
            x_target_kwsidf, \
            y_target,  \
            x_target_mask, \
            x_ood_wid,  \
            x_ood_kws, \
            x_ood_kwsidf, \
            x_ood_mask, \
            y_ood


    def get_support_set_one_hot(self, support_set, classe_list):
        cls_id_map = dict()
        for lid in classe_list:
                    cls_id_map[lid] = len(cls_id_map)

        support_set_one_hot = numpy.zeros([len(support_set), 
                                          len(support_set[0]),
                                          len(cls_id_map)])
        for k in range(len(support_set)):
            for j in range(len(support_set[k])):
                support_set_one_hot[k][j][cls_id_map[support_set[k][j]]] = 1.0

        return support_set_one_hot

    def get_one_hot(self, y_target, classe_list):
        cls_id_map = dict()
        for lid in classe_list:
                    cls_id_map[lid] = len(cls_id_map)

        y_target_one_hot = numpy.zeros([len(y_target), len(cls_id_map)])
        for k in range(len(y_target)):
            y_target_one_hot[k][cls_id_map[y_target[k]]] = 1.0
        return y_target_one_hot


    def _build_pairs(self):

        examples = []
        
        #bs = []
        #for _ in range(10):
        #    bs.extend(np.random.permutation(range(len(self.data))).tolist())
        bs = np.random.permutation(range(len(self.data))).tolist()

        print("len batch episode training dat: ", len(bs))
        sys.stdout.flush()

        for b in bs:
        #for b in numpy.random.permutation(range(len(self.data))):

            self.X_sup = []
            self.Y_sup = [] 
            self.X_sup_mask = []
            self.X_q = [] 
            self.Y_q = [] 
            self.X_q_mask = []
            self.X_neg = [] 
            self.X_neg_mask = []
            self.Y_neg = []
            self.Y_sup_onehot = []
            self.Y_q_onehot = []
            self.Y_neg_onehot = []
            self.X_sup_len = []
            self.Xq_len = []
            self.X_neg_len = []
                 
            curr_workspace = self.data[b]
            ood_wordspace = self.data[:b] + self.data[b+1:]

            target_sets_files = curr_workspace.target_sets_files

            X_sup, KWS_sup, KWSidf_sup, Y_sup, X_sup_mask, \
            X_q, KWS_q, KWSidf_q, Y_q, X_q_mask, \
            X_neg, KWS_neg, KWSidf_neg, X_neg_mask, Y_neg = \
            self.get_train_batch(curr_workspace, ood_wordspace)
            

            Y_sup_onehot = self.get_support_set_one_hot(Y_sup, range(10))
            Y_q_onehot = self.get_one_hot(Y_q, range(10))
            Y_neg_onehot = self.get_one_hot(Y_neg, range(10))
           
            X_sup_len_ = []
            for sup in X_sup:
                lens = []
                for d in sup:
                    lens.append(np.count_nonzero(d))
                X_sup_len_.append(lens)

            Xq_len_ = []
            for sup in X_q:
                len_ = np.count_nonzero(sup)
                Xq_len_.append([len_])

            X_neg_len_ = []
            for sup in X_neg:
                len_ = np.count_nonzero(sup)
                X_neg_len_.append([len_])

            self.X_sup = X_sup
            self.KWS_sup = KWS_sup
            self.KWSidf_sup = KWSidf_sup
            self.Y_sup = Y_sup
            self.X_sup_mask = X_sup_mask
            self.X_q = X_q
            self.KWS_q = KWS_q
            self.KWSidf_q = KWSidf_q
            self.Y_q = Y_q
            self.X_q_mask = X_q_mask
            self.X_neg = X_neg
            self.KWS_neg = KWS_neg
            self.KWSidf_neg = KWSidf_neg
            self.X_neg_mask = X_neg_mask
            self.Y_sup_onehot = Y_sup_onehot
            self.Y_q_onehot=Y_q_onehot
            self.Y_neg=Y_neg
            self.Y_neg_onehot=Y_neg_onehot
            self.X_sup_len=X_sup_len_
            self.Xq_len=Xq_len_
            self.X_neg_len=X_neg_len_

            self.X_sup = np.array(self.X_sup)
            self.KWS_sup = np.array(self.KWS_sup)
            self.KWSidf_sup = np.array(self.KWSidf_sup)
            self.Y_sup = np.array(self.Y_sup)
            self.X_sup_mask = np.array(self.X_sup_mask)
            self.X_q = np.array(self.X_q)
            self.KWS_q = np.array(self.KWS_q)
            self.KWSidf_q = np.array(self.KWSidf_q)
            self.Y_q = np.array(self.Y_q)
            self.X_q_mask = np.array(self.X_q_mask)
            self.X_neg = np.array(self.X_neg)
            self.KWS_neg = np.array(self.KWS_neg)
            self.KWSidf_neg = np.array(self.KWSidf_neg)
            self.X_neg_mask = np.array(self.X_neg_mask)
            self.Y_sup_onehot = np.array(self.Y_sup_onehot)
            self.Y_q_onehot = np.array(self.Y_q_onehot)
            self.Y_neg = np.array(self.Y_neg)
            self.Y_neg_onehot = np.array(self.Y_neg_onehot)
            self.X_sup_len = np.array(self.X_sup_len)
            self.Xq_len = np.array(self.Xq_len)
            self.X_neg_len = np.array(self.X_neg_len)

                

            example_dict = {'X_sup': torch.from_numpy(self.X_sup.astype(np.int64)), 'KWS_sup': torch.from_numpy(self.KWS_sup.astype(np.int64)), 'KWSidf_sup': torch.from_numpy(self.KWSidf_sup), 'X_sup_len': torch.from_numpy(self.X_sup_len), 'Y_sup': torch.from_numpy(self.Y_sup), 'Y_sup_oh': torch.from_numpy(self.Y_sup_onehot.astype(np.int64)), 'X_q':torch.from_numpy(self.X_q.astype(np.int64)), 'KWS_q':torch.from_numpy(self.KWS_q.astype(np.int64)), 'KWSidf_q':torch.from_numpy(self.KWSidf_q),  'Xq_len':torch.from_numpy(self.Xq_len), 'Y_q': torch.from_numpy(self.Y_q), 'Y_q_oh': torch.from_numpy(self.Y_q_onehot.astype(np.int64)), 'X_neg': torch.from_numpy(self.X_neg.astype(np.int64)), 'KWS_neg': torch.from_numpy(self.KWS_neg.astype(np.int64)),'KWSidf_neg':torch.from_numpy(self.KWSidf_neg), 'X_neg_len': torch.from_numpy(self.X_neg_len), 'Y_neg': torch.from_numpy(self.Y_neg), 'Y_neg_oh': torch.from_numpy(self.Y_neg_onehot.astype(np.int64)), 'target_sets_files': target_sets_files}

            examples.append(example_dict)


        return examples
        
    def __len__(self):
        return len(self.pair_data)

    def __getitem__(self, index):
        return self.pair_data[index]



def train_kw_model(params, model, experiment, optimizer):

    TEST_FILE_INDEX = 2
    DATA_PATH = MAIN_PATH+'/data'
    RSL_PATH = MAIN_PATH+'/results'
    
    train_data = get_data(params,
                        params['training_dir'],
                        params['training_list'],
                        role='train_workspace')

    print("len train_data:", len(train_data))
    
    dev_data = get_data(params,
                         params['mnet_dev_list'],
                         role='valid_workspace')

    print("len dev_data:", len(dev_data))

    test_data = get_data(params,
                         params['testing_list'],
                         role='test_workspace')

    print("len test_data:", len(test_data))
    
    save_pickle(DATA_PATH, 'tr_dev_te_kw_intent.pkl', (train_data, dev_data, test_data)) # k=100
    
    
    best_v_macro_avg_eer = 1.0
    total_epoches = 61

    train_loss_all = []

    for e in range(0, total_epoches):
        avg_loss = 0.0
        it = 0

        train_set = AmazonLoader(train_data, params)
        
        b = numpy.random.permutation(range(len(train_set)))
       
        for id_tr in b:
            id_tr = np.array([id_tr])
            train_sampler = SubsetRandomSampler(id_tr)
            train_dl = DataLoader(train_set, sampler=train_sampler, batch_size=1)
            train_loss, domains = experiment.run_training_kw(params, train_dl, optimizer, e)\

            print('Epoch {}: Domain{}:  Task_loss: {}'.format(e, domains[0], 
              train_loss))

            avg_loss += train_loss
            it +=1

            train_loss_all.append(avg_loss/it)
            if len(train_loss_all) > 1:
                min_train_loss = min(train_loss_all)
                print("Min train_loss so far: ", min_train_loss)
                if (avg_loss/it) <= min_train_loss:
                    print("Saving the best model at epoch:", e)
                    state = {'epoch': e, 'state_dict': model.state_dict(), 'optim_dict' : optimizer.state_dict()}
                    torch.save(state, open(os.path.join(RSL_PATH, 'imax_kw_intent_k100.best.pth'), 'wb')) 
                    
       
        print('Epoch {}: Avg_Task_loss: {}'.format(e,  
              (avg_loss/len(b))))


if __name__ == '__main__':

    EBD_PATH = HOME_DIR+'/embeddings/'
    RSL_PATH= HOME_DIR+'/results/'

    MAIN_PATH = HOME_DIR

    parser = argparse.ArgumentParser(description="Training ProtoInfoMax++ on intent classification.")
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

    voc, w2v = load_w2v(MAIN_PATH)
    word2idx, idx2word = read_pickle(EBD_PATH, 'dict_idx2word_intent.pkl')


    params['vocabulary'] = word2idx
    params['voclist'] = idx2word
    params["wordvectors"] = w2v
    params["word2idx"] = word2idx
    params["idx2word"] = idx2word

    cv, word_count_vector, tfidf_transformer = read_pickle(EBD_PATH, 'tfidf_sparse_vec_intent.pkl')
    params["cv"] = cv
    params["tfidf_transformer"] = tfidf_transformer
   
    model = GRUEncoder(params, w2v, len(voc))
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 1e-3)

    print(model)
            
    experiment = RunExperiment(model, params)

    train_kw_model(params, model, experiment, optimizer)
    