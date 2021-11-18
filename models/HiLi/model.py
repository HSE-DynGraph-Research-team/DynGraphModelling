import torch
import torch.nn as nn
import math
import os
import numpy as np
from torch.nn import functional as F

class Model(nn.Module):
    def __init__(self,
                 user_dim,
                 item_dim,
                 num_users,
                 num_items,
                 num_feats,
                 size
                ):
        super(Model, self).__init__()
        self.size = size
        self.user_dim = user_dim
        self.item_dim = item_dim
        self.user_item_att = nn.Linear(item_dim+1+num_feats,1)
        self.user_user_att = nn.Linear(user_dim,1)
        self.item_user_att = nn.Linear(user_dim+1+num_feats,1)
        self.item_item_att = nn.Linear(item_dim,1)
        self.user_item_emb = nn.Linear(item_dim+1+num_feats,user_dim)
        self.item_user_emb = nn.Linear(user_dim+1+num_feats,item_dim)

        self.user_item_add = nn.Linear(item_dim+1+num_feats,1)
        self.user_user_add = nn.Linear(user_dim,1)
        self.item_self_add = nn.Linear(item_dim,1)
        self.item_item_add = nn.Linear(item_dim,1)

        self.prev_att = nn.Linear(item_dim+num_items,1)
        self.prev_wgt = nn.Linear(size,1)
        pred_dim = item_dim + num_items
        self.pred_emb = nn.Linear(user_dim
                                    +item_dim
                                    +num_users
                                    +num_items,
                                  pred_dim
                                 )
        self.pred_itm = nn.Linear(num_items,pred_dim,bias=False)
        self.pred_usr = nn.Linear(num_users,num_items,bias=False)
        self.pred_lnr = nn.Linear(pred_dim,item_dim+num_items)
        self.pred_dyn = nn.Linear(user_dim+item_dim,item_dim)
        self.dyna_usr = nn.Linear(user_dim,item_dim)
        self.dyna_itm = nn.Linear(item_dim,item_dim,bias=False)
        self.pred_stt = nn.Linear(num_users+num_items,num_items)
        self.time_lnr = nn.Linear(1, user_dim)
        self.pred_itm.weight = nn.Parameter(torch.eye(num_items) / math.sqrt(num_items))
        self.dyna_itm.weight = nn.Parameter(torch.eye(item_dim) / math.sqrt(item_dim))

    def forward(self,
                user_embs=None,
                item_embs=None,
                timediff=None,
                mode='user',
                freq=None,
                user_stat=None,
                item_stat=None,
                item_max=4,
                item_pow=0.75,
                user_max=4,
                user_pow=0.75
               ):
        if mode == 'user':
            # print("\t user")
            item_coe = self.user_item_att(item_embs)
            # print("item_embs", torch.isnan(item_embs).sum())
            # print("item_coe", torch.isnan(item_coe).sum())
            user_coe = self.user_user_att(user_embs)
            # print("user_embs", torch.isnan(user_embs).sum())
            # print("user_coe", torch.isnan(user_coe).sum())
            attn_coe = item_coe * user_coe
            # print("attn_coe", torch.isnan(attn_coe).sum())
            neib_emb = self.user_item_emb((torch.exp(attn_coe - attn_coe.max())*item_embs).sum(0).unsqueeze(0))
            # print("neib_emb", torch.isnan(neib_emb).sum())
            output = F.normalize(neib_emb+user_embs)

        elif mode == 'item':
            # print("\t item")
            item_coe = self.item_item_att(item_embs)
            # print("item_embs", torch.isnan(item_embs).sum())
            # print("item_coe", torch.isnan(item_coe).sum())
            user_coe = self.item_user_att(user_embs)
            # print("user_embs", torch.isnan(user_embs).sum())
            # print("user_coe", torch.isnan(user_coe).sum())
            attn_coe = item_coe * user_coe
            # print("attn_coe", torch.isnan(attn_coe).sum())
            neib_emb = self.item_user_emb((torch.exp(attn_coe - attn_coe.max())*user_embs).sum(0).unsqueeze(0))
            # print("neib_emb", torch.isnan(neib_emb).sum())
            output = F.normalize(neib_emb + item_coe*item_embs)

        elif mode == 'pred':
            freq = freq.clone()
            freq[freq>user_max] = user_max
            freq /= user_max
            freq = (freq**user_pow)*user_max

            dyna_usr = self.dyna_usr(F.normalize(freq.unsqueeze(1)*user_embs[:,:self.user_dim]))
            dyna_itm = self.dyna_itm(user_embs[:,self.user_dim:])
            dyna_pre = dyna_usr + dyna_itm
            user_fre = self.pred_usr(user_stat*freq.unsqueeze(1)).detach()
            user_slf = self.pred_usr(user_stat)
            user_pre = user_fre + user_slf
            stat_pre = item_stat + user_pre
            output = torch.cat([dyna_pre,stat_pre],dim=1)
        elif mode == 'prev':
            freq = freq.clone()
            freq[freq>item_max] = item_max
            freq /= item_max
            freq = (freq**item_pow)*item_max
            times = np.array([1]*self.size)
            times[-1]=2
            times = torch.tensor(times).float()
            freq = freq*times
            stat_len = freq.sum(1).unsqueeze(1)
            stat_len = 1/stat_len
            zero_idx = torch.isinf(stat_len)
            stat_len[zero_idx] = 1
            fre = freq.clone()
            fre[torch.nonzero(zero_idx)[:,0],-1] = 1

            mask = F.normalize(fre.unsqueeze(2),dim=2)
            prev_emb = mask * user_embs
            output = F.normalize(prev_emb.mean(1))
        elif mode == 'stat':
            freq = freq.clone()
            freq[freq>item_max] = item_max
            freq /= item_max
            freq = (freq**item_pow)
            times = np.array([1]*self.size)
            times[-1]=2
            times = torch.tensor(times).float()
            freq = freq* times
            stat_len = freq.sum(1).unsqueeze(1)
            stat_len = 1/stat_len
            zero_idx = torch.isinf(stat_len)
            stat_len[zero_idx] = 1
            fre = freq.clone()
            fre[torch.nonzero(zero_idx)[:,0],-1] = 1

            mask = F.normalize(fre.unsqueeze(2),dim=2)
            stat_mean = (mask*fre.unsqueeze(2)*item_stat)
            stat_pre = 1*self.pred_itm(stat_mean[:,-1,:].squeeze(1))
            stat_his = self.pred_itm(stat_mean[:,:-1,:]).sum(1).detach()
            output = stat_pre + stat_his
        elif mode == 'time':
            pass
        elif mode == 'addu':
            item_coe = self.user_item_add(item_embs)
            user_coe = self.user_user_add(user_embs)
            attn_coe = item_coe + user_coe
            neib_emb = self.user_item_emb((attn_coe*item_embs).sum(0))
            output = F.normalize(neib_emb
                                    + 2*user_coe*user_embs)
        elif mode == 'addi':
            freq = freq.clone()
            freq[freq>item_max] = item_max
            freq /= item_max
            freq = (freq**item_pow)*item_max
            times = np.array([1]*self.size)
            times[-1]=2
            times = torch.tensor(times).float()
            freq = freq* times
            mask = F.normalize(freq.unsqueeze(2),dim=2)
            item_coe = self.item_item_att(item_embs).detach()
            self_coe = (self.item_item_att(user_embs)).unsqueeze(1)
            attn_coe = item_coe * self_coe
            neib_emb = freq.unsqueeze(2)*torch.exp(attn_coe - attn_coe.max())*user_embs.unsqueeze(1)
            output = F.normalize(mask*(neib_emb
                                    + (item_embs)),
                                dim=2) + (1-mask)*item_embs
        return output


class Model2(nn.Module):
    def __init__(self,user_dim,item_dim,num_users,num_items,num_feats):
        super(Model2,self).__init__()
        self.user_rnn = nn.GRUCell(item_dim+1+num_feats,user_dim)
        self.item_rnn = nn.GRUCell(user_dim+1+num_feats,item_dim)
        self.pred_lnr = nn.Linear(user_dim
                                    +item_dim
                                    +num_users
                                    +num_items,
                                    #+num_feats,
                                  item_dim
                                    +num_items
                                 )
        self.time_lnr = nn.Linear(1,user_dim)

    def forward(self,user_embs,item_embs=None,time_diff=None,mode='user'):
        if mode == 'user':
            output = F.normalize(self.user_rnn(item_embs,user_embs))
        elif mode == 'item':
            output = F.normalize(self.item_rnn(user_embs,item_embs))
        elif mode == 'pred':
            output = self.pred_lnr(user_embs)
        elif mode == 'time':
            output = self.time_lnr(time_diff)*user_embs
        return output

class ModelNtNs(nn.Module):
    def __init__(self,user_dim,item_dim):
        super(ModelNtNs,self).__init__()
        self.user_rnn = nn.GRUCell(user_dim,user_dim)
        self.item_rnn = nn.GRUCell(item_dim,item_dim)
        self.pred_lnr = nn.Linear(user_dim+item_dim,
                                  item_dim
                                 )

    def forward(self,user_embs,item_embs=None,time_diff=None,mode='user'):
        if mode == 'user':
            output = self.user_rnn(item_embs,user_embs)
        elif mode == 'item':
            output = self.item_rnn(user_embs,item_embs)
        elif mode == 'pred':
            output = self.pred_lnr(user_embs)
        return output


if __name__ == '__main__':
    model = Model(5,5)
    user_embs = torch.rand(1,5)
    user_embs = user_embs.repeat(5,1)
    item_embs = torch.rand(1,5)
    item_embs = item_embs.repeat(5,1)
    user_embs = model(user_embs,item_embs,'user')
    item_embs = model(user_embs,item_embs,'item')
    pass
