import os
import sys
sys.path.append(os.getcwd())
#from __future__ import print_function
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from workspace_cls import SENT_WORDID, SENT_LABELID, SENT_WORD_MASK, SENT_ORIGINAL_TXT
import numpy
import random
import os
import sys
sys.path.append(os.getcwd())
from torch.utils.data import Dataset, DataLoader

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



class RunExperiment:

    def __init__(self, model, params):
        self.model = model
        self.params = params

    def run_training_epoch(self, params, train_dl, optimizer, epoch):

        RSL_PATH= HOME_DIR+'/results'
        
        model = self.model
        idx2word = params['idx2word']
        
        total_loss = 0.
        i = 0
        domains = []
        for b in train_dl:

            x, x_len, y, y_oh, xq, xq_len, yq, yq_oh, x_ood, x_ood_len, y_ood, y_ood_oh, domain = b['X_sup'], b['X_sup_len'], b['Y_sup'], b['Y_sup_oh'], b['X_q'], b['Xq_len'], b['Y_q'], b['Y_q_oh'], b['X_neg'], b['X_neg_len'], b['Y_neg'], b['Y_neg_oh'], b['target_sets_files']

            x = x.squeeze()
            x_len = x_len.squeeze()
            y = y.squeeze()
            y_oh = y_oh.squeeze()
            xq = xq.squeeze()
            xq_len = xq_len.squeeze()
            yq = yq.squeeze()
            yq_oh = yq_oh.squeeze()
            x_ood = x_ood.squeeze()
            x_ood_len = x_ood_len.squeeze()
            y_ood = y_ood.squeeze()
            y_ood_oh = y_ood_oh.squeeze()

            x_ = x
            y_ = y
            xq_ = xq
            yq_ = yq
            x_ood_ = x_ood
            y_ood_ = y_ood

            
            x = x.view(200 * self.params['min_ss_size'], self.params['max_length'])
            x_len = x_len.view(200 * self.params['min_ss_size'])
            y = y.view(200 * self.params['min_ss_size'])
            y_oh = y_oh.view(200 * self.params['min_ss_size'], 2)
            
           

            loss = model(x, x_len, y_oh, xq, xq_len, yq_oh, x_ood, x_ood_len, y_ood_oh)
            
            loss.backward()
            optimizer.step()
            domains.append(domain)

            total_loss += loss.item()

            i+=1

        train_loss = total_loss/i

        if epoch % 10 == 0:
            print(train_loss)
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optim_dict' : optimizer.state_dict()}
            torch.save(state, open(os.path.join(RSL_PATH, 'imax_sentiment_k100_%s.pth'%epoch), 'wb'))
          


        return train_loss, domains

    def run_testing_epoch(self, params, dev_dl):

        
        model = self.model
        idx2word = params['idx2word']

        with torch.no_grad():

            preds = []
            all_dataset = []
            probs = []
            gts = []
            avg_conf_ood = []
            for dat in dev_dl:
                for b in dat:
                    x, x_len, y, y_oh, xq, xq_len, yq, dataset = b['X_sup'], b['X_sup_len'], b['Y_sup'], b['Y_sup_oh'], b['X_q'], b['X_q_len'], b['Y_q'], b['target_set_file']

                    x = x.squeeze(0)
                    x_len = x_len.squeeze(0)
                    y = y.squeeze(0)
                    y_oh = y_oh.squeeze(0)

                    xq = xq.squeeze(0)
                    
                    # sorting examples based on classes
                    srt = torch.sort(y, axis=0)
                    id_srt = srt[1][:,0]

                    x = x[id_srt]
                    x_len = x_len[id_srt]
                    y = y[id_srt]
                    y_oh = y_oh[id_srt]

                    x_cpu = x.cpu().numpy()
                    y_ = y.cpu().numpy()

                    x_str = [[idx2word[i] for i in tknids if i in idx2word and idx2word[i] != '</s>'] for tknids in x_cpu[:,0,:]]
                    for eid, (s, ys) in enumerate(zip(x_str, y_[:,0])):
                        s_ = ' '.join(s)
                        s_ = s_.replace('</s>', '').strip()

                    xq_cpu = xq.cpu().numpy()

                    xq_str = [[idx2word[i] for i in tknids if i in idx2word and idx2word[i] != '</s>'] for tknids in xq_cpu]
                    xq_str = ' '.join(xq_str[0])
                        
                    pred = model._predict(x, x_len, y_oh, xq, xq_len)
                    pred = pred.cpu().data.numpy()
                    pred_cls = numpy.argmax(pred)
                    conf = numpy.max(pred)
                    pred_cls_ = ''
                    yq_str = ''
                    if pred_cls == 0:
                        pred_cls_ = '1'
                    else:
                        pred_cls_ = '2'

                    if yq.cpu().data.numpy().tolist()[0][0] == 0:
                        yq_str = '1'
                        probs.append(pred)
                        gts.append(yq.cpu().data.numpy().tolist()[0][0])

                    elif yq.cpu().data.numpy().tolist()[0][0] == 1:
                        yq_str = '2'
                        probs.append(pred)
                        gts.append(yq.cpu().data.numpy().tolist()[0][0])

                    else:
                        yq_str = 'UNCONFIDENT_INTENT_FROM_SLAD'
                        avg_conf_ood.append(conf)
                        probs.append(pred)
                        gts.append(yq_str)
                   
                    atuple = (pred_cls_, yq_str, conf)
                   

                    preds.append(atuple)
                    all_dataset.append(dataset)
                    
        probs = numpy.array(probs)
        gts = numpy.array(gts)

      

        avg_conf_ood = numpy.mean(avg_conf_ood)

        return preds, all_dataset, probs, gts, avg_conf_ood

    def project_data(self, idx_, dev_dl, idx2word, epoch, str_):

        RSL_PATH= HOME_DIR+'/encodeds/imax'

        model = self.model

        with torch.no_grad():

            for i, dat in enumerate(dev_dl):

                for j, sdat in enumerate(dat):

                    x, x_len, y, y_oh, xq, xq_len, yq, dataset = sdat['X_sup'], sdat['X_sup_len'], sdat['Y_sup'], sdat['Y_sup_oh'], sdat['X_q'], sdat['X_q_len'], sdat['Y_q'], sdat['target_set_file']

                    print("dataset:", dataset)
                  

                    x = x.squeeze(0)
                    x_len = x_len.squeeze(0)
                    y = y.squeeze(0)
                    y_oh = y_oh.squeeze(0)

                    xq = xq.squeeze(0)
                    xq_len = xq_len.squeeze(0)

                    # sorting examples based on classes
                    srt = torch.sort(y, axis=0)
                    id_srt = srt[1][:,0]

                    x = x[id_srt]
                    x_len = x_len[id_srt]
                    y = y[id_srt]
                    y_oh = y_oh[id_srt]


                    sims, enc_prototype, x_sup_enc, y_sup, x_q_enc, y_q , x_sup_raw, xq_raw, y_raw_sup, y_raw, x_raw_sup_len, x_raw_q_len = model._encode(x, x_len, y, xq, xq_len, yq)
                    

                    encodeds = sims, enc_prototype, x_sup_enc, y_sup, x_q_enc, y_q, x_sup_raw, xq_raw, y_raw_sup, y_raw, x_raw_sup_len, x_raw_q_len

                    save_pickle(RSL_PATH, 'encoded_sent_imax_k100_%s_%s_%s_%s_%s.pkl'%(str_, epoch, idx_, i, j), (dataset, encodeds))
                    

        return 0

    def project_data_continue(self, i, dat, domain, idx2word, epoch, str_):

        RSL_PATH= HOME_DIR+'/encodeds/imaxg'

        model = self.model

        with torch.no_grad():

            x, xq, y, yq, x_len, xq_len = dat
            

            x = x.squeeze(0)
            x_len = x_len.squeeze(0)
            y = y.squeeze(0)

            
            srt = torch.sort(y, axis=0)
            id_srt = srt[1][:,0]

            x = x[id_srt]
            x_len = x_len[id_srt]
            y = y[id_srt]


            sims, enc_prototype, x_sup_enc, y_sup, x_q_enc, y_q , x_sup_raw, xq_raw, y_raw_sup, y_raw, x_raw_sup_len, x_raw_q_len = model._encode(x, x_len, y, xq, xq_len, yq)

            encodeds = sims, enc_prototype, x_sup_enc, y_sup, x_q_enc, y_q, x_sup_raw, xq_raw, y_raw_sup, y_raw

            save_pickle(RSL_PATH, 'encoded_sent_imax_k100_%s_%s_0_%s.pkl'%(str_, epoch, i), (domain, encodeds))


        return 0

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

