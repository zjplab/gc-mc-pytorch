import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
from torchvision import models
from layers import *
from metrics import rmse, softmax_accuracy, softmax_cross_entropy, mae


class GAE(nn.Module):
    '''
    num_side_features: dim of features 
    '''
    def __init__(self, num_users, num_items, num_classes, num_side_features, nb,
                       u_features, v_features, u_features_side, v_features_side,
                       input_dim, emb_dim, hidden, dropout, encoder_dropout, **kwargs):
        super(GAE, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.dropout = dropout
        self.encoder_dropout=encoder_dropout

        self.u_features = u_features
        self.v_features = v_features
        self.u_features_side = u_features_side
        self.v_features_side = v_features_side

        self.gcl1 = StackGCN(input_dim, hidden[0],
                                    num_users, num_items,
                                    num_classes, torch.relu, self.dropout, bias=True)
        self.denseu1 = Dense(num_side_features, emb_dim, bias=True, dropout=dropout)
        self.denseu2 = Dense(hidden[0]+emb_dim, hidden[1], bias=True, dropout=dropout, act= \
            lambda x:x)
        self.densev1 = Dense(num_side_features, emb_dim, bias=True, dropout=dropout)
        self.densev2 = Dense(hidden[0]+emb_dim, hidden[1], bias=True, dropout=dropout, act= \
            lambda x:x)
        self.bilin_dec = BilinearMixture(num_users=num_users, num_items=num_items,
                                         num_classes=num_classes,
                                         input_dim=hidden[1],
                                         nb=nb, dropout=0)

    def forward(self, u, v, r_matrix):
        '''
        Returns:
        - output:raw output
        - loss
        - rmse_loss
        - mae_loss 
        '''
        u_z, v_z = self.gcl1(self.u_features, self.v_features,
                             range(self.num_users), range(self.num_items), r_matrix)
        #range(self.number) has no use

        u_f = self.denseu1(self.u_features_side[u])
        v_f = self.densev1(self.v_features_side[v])

        input_u=torch.cat((u_z, u_f), dim=1)
        input_v=torch.cat((v_z, v_f), dim=1)
        u_h = self.denseu2(input_u)
        v_h = self.densev2(input_v)

        
        output, m_hat = self.bilin_dec(u_h, v_h, u, v)

        r_mx = r_matrix.index_select(1, u).index_select(2, v)
        loss = softmax_cross_entropy(output, r_mx.float())
        rmse_loss = rmse(m_hat, r_mx.float())
        mae_loss=mae(m_hat, r_mx.float())
        return output, loss, rmse_loss, mae_loss

    def predict(self, u, v, r_matrix):
        '''
        Returns:
        - output:raw output
        - loss
        - rmse_loss
        - mhat: predicted rating 
        '''
        u_z, v_z = self.gcl1(self.u_features, self.v_features,
                             range(self.num_users), range(self.num_items), r_matrix)
        #range(self.number) has no use

        u_f = torch.relu(self.denseu1(self.u_features_side[u]))
        v_f = torch.relu(self.densev1(self.v_features_side[v]))


        #debug
        """ print(u_z.size(), self.weight_u.size(), \
            u_f.size(), self.weight2_u.size(), \
                 v_z.size(),self.weight_v.size(),\
                      v_f.size(), self.weight2_v.size()) """
        u_h=torch.relu( torch.mm(F.dropout(u_z,p=self.dropout), self.weight_u ) + \
            torch.mm(F.dropout(u_f, p=self.dropout), self.weight2_u) )
        v_h=torch.relu(torch.mm( F.dropout(v_z, p=self.dropout), self.weight_v) + \
            torch.mm(F.dropout(v_f, p=self.dropout), self.weight2_v) )
        output, m_hat = self.bilin_dec(u_h, v_h, u, v)

        r_mx = r_matrix.index_select(1, u).index_select(2, v)
        loss = softmax_cross_entropy(output, r_mx.float())
        rmse_loss = rmse(m_hat, r_mx.float())

        return output, loss, rmse_loss, m_hat
