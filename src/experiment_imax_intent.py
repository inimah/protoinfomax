import os
import sys
sys.path.append(os.getcwd())
#from __future__ import print_function
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from workspace.workspace_intent import SENT_WORDID, SENT_LABELID, SENT_WORD_MASK, SENT_ORIGINAL_TXT
import numpy
import numpy as np
import random
import os
from torch.utils.data import Dataset, DataLoader



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

            
            # bs =100

            x = x.view(1000 * self.params['min_ss_size'], self.params['max_length'])
            x_len = x_len.view(1000 * self.params['min_ss_size'])
            y = y.view(1000 * self.params['min_ss_size'])
            y_oh = y_oh.view(1000 * self.params['min_ss_size'], 10)

           

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
            torch.save(state, open(os.path.join(RSL_PATH, 'imax_intent_k100_%s.pth'%epoch), 'wb'))
           

        return train_loss, domains

    def run_testing_epoch(self, params, dev_dl):

       
        model = self.model
        idx2word = params['idx2word']

        with torch.no_grad():

            preds_info = []
            all_dataset = []
            probs = []
            gts = []
            avg_conf_ood = []
            for dat in dev_dl:

                for b in dat:

                    x, x_len, y, y_oh, xq, xq_len, yq, dataset = b['X_sup'], b['X_sup_len'], b['Y_sup'], b['Y_sup_oh'], b['X_q'], b['X_q_len'], b['Y_q'], b['target_set_file']

                    x = x.squeeze()
                    x_len = x_len.squeeze()
                    y = y.squeeze()
                    y_oh = y_oh.squeeze()

                    xq = xq.squeeze(0)

                    x_cpu = x.cpu().numpy()
                    y_ = y.cpu().numpy()
                    

                    xq_cpu = xq.cpu().numpy()
                    xq_cpu = xq_cpu.reshape((xq_cpu.shape[-1]))

                    xq_str = [idx2word[i] for i in xq_cpu if i in idx2word and idx2word[i] != '</s>']
                    xq_str = ' '.join(xq_str)
                    
                    pred = model._predict(x, x_len, y_oh, xq, xq_len)
                    pred = pred.cpu().data.numpy()
                    pred_cls = numpy.argmax(pred)
                    conf = numpy.max(pred)
                    pred_cls_ = ''
                    yq_str = ''
                    if pred_cls==0:
                        pred_cls_ = str(pred_cls)
                    else:
                        pred_cls_ = 'oos'


                   
                    if yq.cpu().data.numpy().tolist()[0][0] ==0:

                        yq_str = str(yq.cpu().data.numpy().tolist()[0][0])
                        probs.append(pred)
                        gts.append(yq.cpu().data.numpy().tolist()[0][0])                        

                    else:

                        yq_str = 'oos'
                        probs.append(pred)
                        gts.append(yq_str)
                        avg_conf_ood.append(conf)
                        


                    atuple = (pred_cls_, yq_str, conf)
                    
                    preds_info.append(atuple)
                    all_dataset.append(dataset)
                    
        probs = numpy.array(probs)
        gts = numpy.array(gts)

        avg_conf_ood = numpy.mean(avg_conf_ood)

        return preds_info, all_dataset, probs, gts, avg_conf_ood

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

