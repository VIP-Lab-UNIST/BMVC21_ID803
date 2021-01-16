import torch
import shutil 
import os
import numpy as np
from functools import reduce
from glob import glob


class MPLP(object):

    def __init__(self, use_coap, use_uniq, use_cycle, total_scene, t, t_c, s_c, r):
        self.cnt2snum = total_scene
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
                if len(co_vec)==0: pass
                else:
                    co_sims = co_vec.mm(memory.t())
                    co_sim = torch.max(co_sims, dim=0)[0]
                    co_sim[co_sim < self.t_c] = 0
                    co_sim *= self.s_c
                    # sum by scene
                    co_sim_sum = torch.zeros((len(self.cnt2snum.unique()), )).cuda().index_add(dim=0, index=self.cnt2snum, source=co_sim)
                    co_sim_sum = torch.gather(input=co_sim_sum, dim=0, index=self.cnt2snum).clamp(max=self.s_c*len(co_vec))
                    sim = (sim+co_sim_sum).clamp(max=1.)

            sim[target] = 1. 
            simsorted, idxsorted = torch.sort(sim ,dim=0, descending=True)
            
            ## UNIQUENESS: Select candidate(top k)
            if self.use_uniq:
                topk_sim=[]
                topk_idx=[]
                topk_scn=[]
                for j, (sim_, idx_) in enumerate(zip(simsorted, idxsorted)):
                    if (self.cnt2snum[idx_] in topk_scn) & (sim_ != 1.).item(): continue
                    if (sim_ < self.t): break
                    topk_sim.append(sim_)
                    topk_scn.append(self.cnt2snum[idx_])
                    topk_idx.append(idx_)
                topk_sim = torch.tensor(topk_sim).cuda()
                topk_idx = torch.tensor(topk_idx).cuda()
                num_topk = len(topk_scn)
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
            neg_idices.append(torch.zeros(len(sim)).scatter_(0, torch.tensor(hn_idxs), 1.).cuda())

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
        path = './logs/outputs/hd/'
        # path = './logs/outputs/tmp/'
        print('path: ', path)
        flist=glob('./logs/outputs/all/*.jpg')
        for i, (target, multilabel) in enumerate(zip(targets, multilabels)):
            fname=flist[target].split('/')[-1].split('.')[0]
            os.makedirs(path+fname)
            for label in (multilabel==-1).nonzero():
                shutil.copy(flist[label], path+fname)
        raise ValueError
