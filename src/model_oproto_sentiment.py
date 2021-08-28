import os
import sys
sys.path.append(os.getcwd())
import torch
from torch import nn
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GRUEncoder(nn.Module):

    def __init__(self, params, w2v, vocab_size:int):
        super().__init__()
        self.params = params
        self.vocab_size = vocab_size
        self.ent_loss = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(0.25)
        self.embed = nn.Embedding(self.vocab_size, self.params['emb_size'])
        self.embed.weight.data.copy_(torch.from_numpy(w2v))
        self.embed.weight.requires_grad = True
        self.embed_dropout = nn.Dropout(p=0.25)
        self.dense_dropout = nn.Dropout(p=0.25)
        self.encoder = nn.GRU(self.params['emb_size'], self.params['hidden_size'], batch_first=True, bidirectional=True)
        self.fc_layer = nn.Linear(2*self.params['hidden_size'], self.params['hidden_size'])
        self.linear = nn.Linear(self.params['hidden_size'], 2)
        self.relu = nn.ReLU()

        # For Attention
        # multiple contexts r = 5
        self.r = 5
        self.head = nn.Parameter(torch.Tensor(self.params['hidden_size'], self.r).uniform_(-0.1, 0.1))
        self.proj = nn.Linear(2*self.params['hidden_size'], self.params['hidden_size'])

    
    def custom_softmax(self, input, axis=1):

        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size)-1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = F.softmax(input_2d)
        soft_max_nd = soft_max_2d.view(*trans_size)

        return soft_max_nd.transpose(axis, len(input_size)-1)

         
    def attention_net(self, x, text_len):

        '''
            text:     batch, max_text_len, input_dim
            text_len: batch, max_text_len
        '''
        if torch.cuda.is_available():
            text_len = torch.LongTensor(text_len).cuda()
        else:
            text_len = torch.LongTensor(text_len)
     
        batch_size, max_text_len, _ = x.size()

        proj_x = torch.tanh(self.proj(x.view(batch_size * max_text_len, -1)))
        att = torch.mm(proj_x, self.head) # batch_size * max_text_len, 5
        att = att.view(batch_size, max_text_len, self.r)  # unnormalized

        # create mask
        if torch.cuda.is_available():
            idxes = torch.arange(max_text_len, out=torch.cuda.LongTensor(max_text_len, device=device)).unsqueeze(0)
        else:
            idxes = torch.arange(max_text_len, out=torch.LongTensor(max_text_len, device=device)).unsqueeze(0)
        mask = (idxes < text_len.unsqueeze(1)).bool()

        att[~mask] = float('-inf')

        # apply softmax
        att_norm = self.custom_softmax(att, 1)  # batch, max_text_len, r
        att_norm = att_norm.transpose(1,2)

        return att, att_norm
    
    def _compute_cos(self, XS, XQ):
       
      

        if XS.ndim == 2:
            XS = XS.unsqueeze(1)
        if XQ.ndim == 2:
            XQ = XQ.unsqueeze(1)

        XQ = F.normalize(XQ, p=2, dim=-1)
        XS = F.normalize(XS, p=2, dim=-1)

        dist=torch.matmul(XQ, XS.permute(0,2,1))

        return dist

    def forward(self, x_sup, x_sup_len, y_sup, y_sup_oh, x_q, x_q_len, y_q, y_q_oh, x_neg, x_neg_len, y_neg):
       
        x_sup = x_sup.squeeze()
        x_sup_len = x_sup_len.squeeze()
        y_sup = y_sup.squeeze()
        y_sup_oh = y_sup_oh.squeeze()
        
        bs_sup = x_sup.size(0) # batch size
        l_sup = [self.params['max_length']] * bs_sup
        bs_q = x_q.size(0)
        l_q = [self.params['max_length']] * bs_q
        bs_neg = x_neg.size(0) # batch size
        l_q_neg = [self.params['max_length']] * bs_neg
        
        x_sup = self.embed_dropout(self.embed(x_sup))
        x_q = self.embed_dropout(self.embed(x_q))
        x_neg = self.embed_dropout(self.embed(x_neg))

       

        # support
        x_sup_len = x_sup_len.cpu().data.numpy().tolist()
        x_sup = pack_padded_sequence(x_sup, l_sup, batch_first=True, enforce_sorted=False)

        x_sup, _ = self.encoder(x_sup) 
        x_sup, _ = pad_packed_sequence(x_sup, batch_first=True) 
        _, alpha_sup = self.attention_net(x_sup, x_sup_len)
        hs_sup = torch.bmm(alpha_sup, x_sup)
        sebd_sup = torch.max(hs_sup, dim=1, keepdim=False)

        x_sup_enc = self.fc_layer(sebd_sup[0])

        # target query
        x_q_len = x_q_len.cpu().data.numpy().tolist()
        x_q = pack_padded_sequence(x_q, l_q, batch_first=True, enforce_sorted=False)
        x_q, _ = self.encoder(x_q) 
        x_q, _ = pad_packed_sequence(x_q, batch_first=True) 
        _, alpha_q = self.attention_net(x_q, x_q_len)
        hs_q = torch.bmm(alpha_q, x_q)
        sebd_q = torch.max(hs_q, dim=1, keepdim=False)

        x_q_enc = self.fc_layer(sebd_q[0])

        # neg.
        x_neg_len = x_neg_len.cpu().data.numpy().tolist()
        x_neg = pack_padded_sequence(x_neg, l_q_neg, batch_first=True, enforce_sorted=False)
        x_neg, _ = self.encoder(x_neg) 
        x_neg, _ = pad_packed_sequence(x_neg, batch_first=True) 
        _, alpha_neg = self.attention_net(x_neg, x_neg_len)
        hs_neg = torch.bmm(alpha_neg, x_neg)
        sebd_neg = torch.max(hs_neg, dim=1, keepdim=False)

        x_neg_enc = self.fc_layer(sebd_neg[0])

        x_sup_enc = x_sup_enc.contiguous().view(200, self.params['min_ss_size'], -1)
        y_sup = y_sup.contiguous().view(200, self.params['min_ss_size'], -1)

      
        # encoded_prototype.shape(B, Y, E)
        encoded_prototype = torch.mean(x_sup_enc, dim=1)

        if encoded_prototype.ndim == 1:
            encoded_prototype = encoded_prototype.unsqueeze(0)
        else:
            encoded_prototype = encoded_prototype.squeeze()
        
        # Compute g(x, y)

        x_q_enc = torch.cat((x_q_enc, x_q_enc), dim=0)
        y_q = torch.cat((y_q, y_q), dim=0)
        y_q_oh = torch.cat((y_q_oh, y_q_oh), dim=0)
        
        bs = x_q_enc.size(0)
     
        logits = self._compute_cos(encoded_prototype.float(), x_q_enc.float()) 
        logits = logits.squeeze()
        logits = logits.unsqueeze(-1)
        
        logits_soft = self.params['softmax_factor'] * logits
        

        mean_logits = torch.mean(logits, dim=0)
        mean_logits_soft = torch.mean(logits_soft, dim=0)

        #logits = logits * y_q_oh
        logits_soft = logits_soft * y_q_oh

       
        labels = y_q
        labels = labels.squeeze(-1)

        # LOSS-1
        #logloss_ontopic = self.ent_loss(logits.to(device), labels.long().to(device))
        logloss_ontopic = self.ent_loss(logits_soft.to(device), labels.long().to(device))
        logloss_ontopic = logloss_ontopic.unsqueeze(-1)

        if epoch%10 ==0:
            print("logloss_ontopic :", logloss_ontopic)
            sys.stdout.flush()

        # LOSS-2
       
        hingeloss_ontopic = torch.max(torch.tensor([0.0]), self.params['ood_threshold']
                                                 + self.params['ood_threshold_margin'] 
                                                 -mean_logits)[0]
        hingeloss_ontopic = hingeloss_ontopic.to(device)


        # Compute 1 - g(x, y^(\bar))
        # OOD query
        # LOSS-3

        x_neg_enc = torch.cat((x_neg_enc, x_neg_enc), dim=0)


        bs_ood, _ = x_neg_enc.size(0), x_neg_enc.size(1)

        _logits = self._compute_cos(encoded_prototype.float(), x_neg_enc.float())
        _logits =_logits.squeeze()
        _logits = _logits.unsqueeze(-1)

        mean_logits_ood = torch.mean(_logits, dim=0)

        
        hingeloss_ood = torch.max(torch.tensor([0.0]), mean_logits_ood - self.params['ood_threshold']
                                                 + self.params['ood_threshold_margin'])[0]
        hingeloss_ood = hingeloss_ood.to(device)

        loss = self.params['alpha_indomain'] * logloss_ontopic + self.params['alpha_pos'] * hingeloss_ontopic + self.params['alpha_neg'] * hingeloss_ood
       
        return loss




    def _encode(self, x_raw_sup, x_raw_sup_len, y_raw_sup, x_raw_q, x_raw_q_len, y_raw):

     
        x_test_sup = x_raw_sup
        y_test_sup = y_raw_sup
        x_test_sup_len = x_raw_sup_len
        x_test_q_len = x_raw_q_len

        x_test_sup = x_test_sup.view(1*2*self.params['min_ss_size'], -1)
        x_test_sup_len = x_test_sup_len.view(1*2*self.params['min_ss_size'])
        y_test_sup = y_test_sup.view(1*2*self.params['min_ss_size'], -1)

       
        bs_sup = x_test_sup.size(0) # batch size
        l_sup = [self.params['max_length']] * bs_sup
        bs_q = x_raw_q.size(0)
        l_q = [self.params['max_length']] * bs_q

        x_sup = self.embed_dropout(self.embed(x_test_sup))
        x_q = self.embed_dropout(self.embed(x_raw_q))

        # support
        x_test_sup_len = x_test_sup_len.cpu().data.numpy().tolist()
        x_sup = pack_padded_sequence(x_sup, l_sup, batch_first=True, enforce_sorted=False)
        x_sup, _ = self.encoder(x_sup) 
        x_sup, _ = pad_packed_sequence(x_sup, batch_first=True) 
        _, alpha_sup = self.attention_net(x_sup, x_test_sup_len)
        hs_sup = torch.bmm(alpha_sup, x_sup)
        sebd_sup = torch.max(hs_sup, dim=1, keepdim=False)

        x_sup_enc = self.fc_layer(sebd_sup[0])

        # target query
        x_test_q_len = x_test_q_len.cpu().data.numpy().tolist()
        x_q = pack_padded_sequence(x_q, l_q, batch_first=True, enforce_sorted=False)
        x_q, _ = self.encoder(x_q) 
        x_q, _ = pad_packed_sequence(x_q, batch_first=True) 
        _, alpha_q = self.attention_net(x_q, x_test_q_len)
        hs_q = torch.bmm(alpha_q, x_q)
        sebd_q = torch.max(hs_q, dim=1, keepdim=False)

        x_q_enc = self.fc_layer(sebd_q[0])

        x_sup_enc = x_sup_enc.view(2,self.params['min_ss_size'], -1)
        y_test_sup = y_test_sup.view(2,self.params['min_ss_size'], -1)

       
        encoded_prototype = torch.mean(x_sup_enc, dim=1)
        

        if encoded_prototype.ndim == 1:
            encoded_prototype = encoded_prototype.unsqueeze(0)
        else:
            encoded_prototype = encoded_prototype.squeeze()

        sims = self._compute_cos(encoded_prototype, x_q_enc)
        sims = sims.cpu().data.numpy()

        encoded_prototype = encoded_prototype.cpu().data.numpy()
        x_sup_enc = x_sup_enc.cpu().data.numpy()
        y_test_sup= y_test_sup.cpu().data.numpy()
        x_q_enc = x_q_enc.cpu().data.numpy()
        yq = y_raw.cpu().data.numpy()

        
        return sims, encoded_prototype, x_sup_enc, y_test_sup, x_q_enc, yq, x_raw_sup, x_raw_q, y_raw_sup, y_raw, x_raw_sup_len, x_raw_q_len

    def _predict(self, x_test_sup, x_test_sup_len, y_test_sup, x_test_q, x_test_q_len):

     
        x_test_sup = x_test_sup.squeeze(0)
        y_test_sup = y_test_sup.squeeze(0)
        x_test_sup_len = x_test_sup_len.squeeze(0)
        x_test_q_len = x_test_q_len.squeeze(-1)

        x_test_sup = x_test_sup.view(1*2*self.params['min_ss_size'], -1)
        x_test_sup_len = x_test_sup_len.view(1*2*self.params['min_ss_size'])
        y_test_sup = y_test_sup.view(1*2*self.params['min_ss_size'], -1)
        
        
        bs_sup = x_test_sup.size(0) # batch size
        l_sup = [self.params['max_length']] * bs_sup
        bs_q = x_test_q.size(0)
        l_q = [self.params['max_length']] * bs_q

        x_sup = self.embed_dropout(self.embed(x_test_sup))
        x_q = self.embed_dropout(self.embed(x_test_q))

        # support
        x_test_sup_len = x_test_sup_len.cpu().data.numpy().tolist()
        x_sup = pack_padded_sequence(x_sup, l_sup, batch_first=True, enforce_sorted=False)
        x_sup, _ = self.encoder(x_sup) 
        x_sup, _ = pad_packed_sequence(x_sup, batch_first=True) 
        _, alpha_sup = self.attention_net(x_sup, x_test_sup_len)
        hs_sup = torch.bmm(alpha_sup, x_sup)
        sebd_sup = torch.max(hs_sup, dim=1, keepdim=False)

        x_sup_enc = self.fc_layer(sebd_sup[0])

        # target query
        x_test_q_len = x_test_q_len.cpu().data.numpy().tolist()
        x_q = pack_padded_sequence(x_q, l_q, batch_first=True, enforce_sorted=False)
        x_q, _ = self.encoder(x_q) 
        x_q, _ = pad_packed_sequence(x_q, batch_first=True) 
        _, alpha_q = self.attention_net(x_q, x_test_q_len)
        hs_q = torch.bmm(alpha_q, x_q)
        sebd_q = torch.max(hs_q, dim=1, keepdim=False)

        x_q_enc = self.fc_layer(sebd_q[0])

     
        x_sup_enc = x_sup_enc.view(2,self.params['min_ss_size'], -1)
        y_test_sup = y_test_sup.view(2,self.params['min_ss_size'], -1)

     
        encoded_prototype = torch.mean(x_sup_enc, dim=1)
        

        if encoded_prototype.ndim == 1:
            encoded_prototype = encoded_prototype.unsqueeze(0)
        else:
            encoded_prototype = encoded_prototype.squeeze()


        logits = self._compute_cos(encoded_prototype.float(), x_q_enc.float())
        logits = logits.squeeze()
        logits = logits.unsqueeze(-1)     
        
        pred = logits
        
        return pred
