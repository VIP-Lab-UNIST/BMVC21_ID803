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
        neg_idices = []
        for i, (target, sim) in enumerate(zip( targets_uniq, mem_sim)):

            ## COAPEARANCE
            if self.use_coap:

                # calculate co-appearance similarity
                mask = (self.cnt2snum==self.cnt2snum[target]) & (torch.tensor(range(len(self.cnt2snum))).cuda()!=target)
                co_vec = memory[mask]

                # advantage for distance
                start = min((self.cnt2snum==self.cnt2snum[target]).nonzero())
                co_dist = torch.tensor(self.dist_mat[str(self.cnt2snum[target].item())]).cuda()[target - start].squeeze().cuda()
                co_dist = co_dist[co_dist!=0].unsqueeze(1)

                if len(co_vec)==0: pass
                else:
                    co_sims = co_vec.mm(memory.t())
                    co_sims = co_sims * (1000/co_dist).clamp(min=0., max=1.)
                    co_sim = torch.max(co_sims, dim=0)[0]
                    co_sim[co_sim < self.t_c] = 0
                    co_sim *= self.s_c
                    # sum by scene
                    co_sim_sum = torch.zeros((len(self.cnt2snum.unique()), )).cuda().index_add(dim=0, index=self.cnt2snum, source=co_sim)
                    co_sim_sum = torch.gather(input=co_sim_sum, dim=0, index=self.cnt2snum).clamp(max=self.s_c*len(co_vec))
                    sim = (sim+co_sim_sum).clamp(max=1.)
            sim[target] = 1. 
            simsorted, idxsorted = torch.sort(sim ,dim=0, descending=True)
            snumsorted = self.cnt2snum[idxsorted]
            
            ## UNIQUENESS: Select candidate(top k)
            if self.use_uniq:
                topk_snum = snumsorted[simsorted>=self.t]
                snums, snums_cnt = torch.unique(topk_snum, return_counts=True)
                mask = (topk_snum[..., None]==snums[snums_cnt==1]).any(-1)
                if len(snums[snums_cnt>=2])!=0:
                    for j, snum in enumerate(snums[snums_cnt>=2]):
                        mask[min((topk_snum==snum).nonzero())] = True
                topk_sim = simsorted[simsorted>=self.t][mask]
                topk_idx = idxsorted[simsorted>=self.t][mask]
                num_topk = len(topk_sim)

            else:
                topk_sim = simsorted[simsorted>=self.t]
                topk_idx = idxsorted[simsorted>=self.t]
                num_topk = len(topk_idx)

            ## CYCLE CONSISTENCY
            if self.use_cycle:
                topk_vec = memory[topk_idx]
                topk_sim = topk_vec.mm(memory.t())
                topk_sim_sorted, topk_idx_sorted = torch.sort(topk_sim.detach().clone(), dim=1, descending=True)

                cycle_idx = []
                for j in range(num_topk):
                    pos = torch.nonzero(topk_idx_sorted[j] == target).item()
                    # if pos > max(num_topk, 20): break
                    if pos > max(num_topk, 20): continue
                    cycle_idx.append(topk_idx_sorted[j, 0])
                
                if len(cycle_idx) == 0: multilabel[i, target] = float(1)
                else: 
                    cycle_idx = torch.tensor(cycle_idx).cuda()    
                    multilabel[i, cycle_idx] = float(1)
            else: multilabel[i, topk_idx] = float(1)
            
            ## SELECT HARD NEGATIVE SAMPLE: Candidate view
            pos_vec = memory[topk_idx]
            pos_sims = pos_vec.mm(memory.t())
            hn_idxs_list = []
            for j, pos_sim in enumerate(pos_sims):
                pos_sim[topk_idx] = 0
                neg_simsorted, neg_idxsorted = torch.sort(pos_sim, dim=0, descending=True)
                num = int(self.r * len(pos_sim.nonzero()))
                hn_idxs_list.append(neg_idxsorted[:num].detach().cpu().tolist())
            hn_idxs = set(sum(hn_idxs_list, []))
            topk_idx = set(topk_idx)
            hn_idxs = list(hn_idxs.difference(topk_idx))

            neg_idices.append(torch.zeros(len(sim)).scatter_(0, torch.tensor(hn_idxs).long(), 1.).cuda())

            # multilabel[i, neg_idxs] = float(-1)

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
        # path = './logs/outputs/mAP6/sc{:.2f}_th{:.2f}__/'.format(self.s_c, self.t_c)
        # path = './logs/outputs/mAP6/no_t{:.2f}_sc{:.2f}_tc{:.2f}/'.format(self.t, self.s_c, self.t_c)
        # path = './logs/outputs/mAP6/co_t{:.2f}_sc{:.2f}_tc{:.2f}/'.format(self.t, self.s_c, self.t_c)
        # path = './logs/outputs/mAP6/cy_t{:.2f}_sc{:.2f}_tc{:.2f}/'.format(self.t, self.s_c, self.t_c)
        # path = './logs/outputs/hd/'
        path = './logs/outputs/tmp/'
        print('path: ', path)
        flist=glob('./logs/outputs/all/*.jpg')
        for i, (target, multilabel) in enumerate(zip(targets, multilabels)):
            fname=flist[target].split('/')[-1].split('.')[0]
            os.makedirs(path+fname)
            for label in (multilabel==1).nonzero():
                shutil.copy(flist[label], path+fname)
        raise ValueError
