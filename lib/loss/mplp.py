import torch
import shutil 
import os
import numpy as np
import json

from tqdm import tqdm
from functools import reduce
from glob import glob

class MPLP(object):

    def __init__(self, use_coap, use_uniq, use_cycle, total_scene, t, t_c, s_c, r):
        self.cnt2snum = total_scene
        with open('./dist_dict.json', 'r') as fp:
            self.dist_mat = json.load(fp)
        self.t = t
        self.s_c = s_c
        self.t_c = t_c
        self.r = r
        self.use_coap = use_coap
        self.use_uniq = use_uniq
        self.use_cycle = use_cycle

    def predict(self, memory, targets):

        targets_uniq=targets.unique()
        mem_vec = memory[targets_uniq]
        mem_sim = mem_vec.mm(memory.t())
        
        multilabel = torch.zeros(mem_sim.shape).cuda()
        neg_idices = torch.zeros((targets.shape[0], mem_sim.shape[1])).cuda()
        for i, (target, sim) in enumerate(zip( targets_uniq, mem_sim)):

            ## COAPEARANCE
            if self.use_coap:
                # calculate co-appearance similarity
                mask = (self.cnt2snum==self.cnt2snum[target]) & (torch.tensor(range(len(self.cnt2snum))).cuda()!=target)
                co_vec = memory[mask]
                if len(co_vec)==0: pass
                else:
                    # co_sims = co_vec.mm(memory.t())
                    # co_sim = torch.max(co_sims, dim=0)[0]
                    # sim = torch.tanh(1.1*(sim-co_sim))

                    co_sims = co_vec.mm(memory.t())
                    # co_sims = co_sims * (1000/co_dist).clamp(min=0., max=1.)
                    co_sim = torch.max(co_sims, dim=0)[0]
                    co_sim[co_sim < self.t_c] = 0
                    # co_sim *= self.s_c
                    # sum by scene
                    co_sim_sum = torch.zeros((len(self.cnt2snum.unique()), )).cuda().index_add(dim=0, index=self.cnt2snum, source=co_sim)
                    co_sim_sum = torch.gather(input=co_sim_sum, dim=0, index=self.cnt2snum).clamp(max=self.s_c*len(co_vec))
                    sim = (sim+co_sim_sum).clamp(max=1.)
                    # sim = torch.tanh(1.5*(sim+co_sim_sum))

            sim[target] = 1. 
            simsorted, idxsorted = torch.sort(sim ,dim=0, descending=True)
            snumsorted = self.cnt2snum[idxsorted]
            
            ## UNIQUENESS: Select candidate(top k)
            if self.use_uniq:
                cand_snum = snumsorted[simsorted>=self.t]
                snums, snums_cnt = torch.unique(cand_snum, return_counts=True)
                mask = (cand_snum[..., None]==snums[snums_cnt==1]).any(-1)
                if len(snums[snums_cnt>=2])!=0:
                    for j, snum in enumerate(snums[snums_cnt>=2]):
                        mask[min((cand_snum==snum).nonzero())] = True
                cand_sim = simsorted[simsorted>=self.t][mask]
                cand_idx = idxsorted[simsorted>=self.t][mask]
                num_cand = len(cand_sim)

            else:
                cand_sim = simsorted[simsorted>=self.t]
                cand_idx = idxsorted[simsorted>=self.t]
                num_cand = len(cand_idx)

            ## SCENE CYCLE CONSISTENCY
            if self.use_cycle:
                
                cand_vec = memory[cand_idx]
                cand_sim = cand_vec.mm(memory.t())
                cand_scn = self.cnt2snum[cand_idx]
                cand_co_scn_mask = (self.cnt2snum.unsqueeze(1)==cand_scn).any(-1)
                topk_idx = cand_co_scn_mask.nonzero().squeeze()[torch.topk(cand_sim[:, cand_co_scn_mask], k=len(cand_sim), dim=1)[1]]

                # check cycle consistency
                topk_mask = (topk_idx.sort(dim=1)[0]==cand_idx.sort(dim=0)[0]).all(dim=1)
                cycle_idx = cand_idx[topk_mask]
                
                if len(cycle_idx) == 0: multilabel[i, target] = float(1)
                else: 
                    multilabel[i, cycle_idx] = float(1)
                    
            else: multilabel[i, cand_idx] = float(1)

            ## SELECT HARD NEGATIVE SAMPLE: Candidate view
            # intersection
            pos_vec = memory[cand_idx]
            pos_sims = pos_vec.mm(memory.t())
            pos_sims[:,cand_idx] = -1
            neg_simsorted, neg_idxsorted = torch.sort(pos_sims, dim=1, descending=True)
            num = int(self.r * len((pos_sims[0]!=-1).nonzero()))
            hn_idxs = neg_idxsorted[:,0:num].reshape(-1)
            hn_idxs, hn_idxs_cnt = torch.unique(hn_idxs, return_counts=True)
            hn_idxs = hn_idxs[hn_idxs_cnt == len(cand_idx)]
            # hn_idxs = hn_idxs[hn_idxs_cnt >= len(cand_idx)*1.]

            neg_idices[i] = torch.zeros(len(sim)).scatter_(0, hn_idxs.detach().cpu(), 1.).cuda()
            # multilabel[i, neg_idices[i]] = float(-1)

        ## Expand multi-label
        multilabel_=torch.zeros((targets.shape[0], multilabel.shape[1])).cuda()
        neg_idices_=torch.zeros((targets.shape[0], multilabel.shape[1])).cuda()
        for t_uniq, mlabel, neg_idx in zip(targets_uniq, multilabel, neg_idices):
            midx = (t_uniq==targets).nonzero()
            multilabel_[midx, :] = mlabel
            neg_idices_[midx, :] = neg_idx
        targets = torch.unsqueeze(targets, 1)
        multilabel_.scatter_(1, targets, float(1))

        # self.draw_proposal(targets_uniq, multilabel)
        # raise ValueError

        return multilabel_, neg_idices_

    def draw_proposal(self, targets, multilabels):
        # path = './logs/outputs/mAP6/no_t{:.2f}_sc{:.2f}_tc{:.2f}/'.format(self.t, self.s_c, self.t_c)
        # path = './logs/outputs/coap/no_coap/'
        # path = './logs/outputs/coap/no_coap_epoch10/'
        path = './logs/outputs/coap/cycle_th05/'
        # path = './logs/outputs/coap/no_prior_th04/'
        print('path: ', path)
        flist=glob('./logs/outputs/all/*.jpg')
        for i, (target, multilabel) in enumerate(zip(targets, multilabels)):
            fname=flist[target].split('/')[-1].split('.')[0]
            os.makedirs(path+fname)
            for label in (multilabel==1).nonzero():
                shutil.copy(flist[label], path+fname)
        raise ValueError
