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
        # self.t = 0.2
        self.s_c = s_c
        self.t_c = t_c
        self.r = r
        self.use_coap = use_coap
        self.use_uniq = use_uniq
        self.use_cycle = use_cycle
        
    def forward_matching(self, sim_forward):
        # Input argument
        #  sim_forward (1D float tensor) : 1-dimensional vector with size of N
        # Output 
        #  cand_sim (1D float tensor) : Top-1 similarity larger than the threshold on each scene
        #  cand_idx (1D long tensor) : person indices Top-1 similarity larger than the threshold on each scene

        simsorted, idxsorted = torch.sort(sim_forward ,dim=0, descending=True)
        snumsorted = self.cnt2snum[idxsorted]
        
        ## Select candidate pairs whose similarities are larger than the Threshold
        selected = simsorted>0
        cand_snum = snumsorted[selected]
        cand_sim = simsorted[selected]
        cand_idx = idxsorted[selected]

        ## Select Top-1 per each scene
        snums, snums_cnt = torch.unique(cand_snum, return_counts=True)
        mask = (cand_snum[..., None]==snums[snums_cnt==1]).any(-1)
        if len(snums[snums_cnt>1])!=0:
            for snum in snums[snums_cnt>1]:
                mask[min((cand_snum==snum).nonzero())] = True
        cand_sim = cand_sim[mask]
        cand_idx = cand_idx[mask]
        
        return cand_sim, cand_idx 

    def sacc(self, sim_forward, query_pid, memory):
        # Scene-Aware Cycle Consistency for "a single query"
        co_persons = (self.cnt2snum==self.cnt2snum[query_pid]).nonzero().squeeze(1)
        co_persons, _ = torch.sort(co_persons, dim=0, descending=False)
        offset = co_persons[0]
        forward_matched_sim, forward_matched_idx  = self.forward_matching(sim_forward)
        if len(forward_matched_idx) > 0:
            sim_backward = memory[forward_matched_idx].mm(memory[co_persons].t())
            backward_matched_idx = sim_backward.max(dim=1)[1]
            cycle_matched_idx = forward_matched_idx[backward_matched_idx==(query_pid-offset)]
            cycle_matched_sim = forward_matched_sim[backward_matched_idx==(query_pid-offset)]
            return cycle_matched_sim, cycle_matched_idx
        else:
            return forward_matched_sim, forward_matched_idx # empty index tensor

    # def squeeze_scene(self, mem_sim):
    #     simsorted, idxsorted = torch.sort(mem_sim ,dim=1, descending=True)
    #     scene_num = max(self.cnt2snum)
    #     mem_sim_squeeze = []
    #     mem_idx_squeeze = []
    #     for simsorted_, idxsorted_ in zip(simsorted, idxsorted):
    #         snumsorted_ = self.cnt2snum[idxsorted_]
    #         print(simsorted_.shape)
    #         ## Select Top-1 per each scene
    #         mask =  torch.zeros_like(simsorted_).bool()
    #         for snum in range(scene_num.item()):
    #             mask[min((snumsorted_==snum).nonzero())] = True

    #         mem_sim_squeeze.append(simsorted_[mask])
    #         mem_idx_squeeze.append(idxsorted_[mask])

    #     mem_sim_squeeze = torch.stack(mem_sim_squeeze)
    #     mem_idx_squeeze = torch.stack(mem_idx_squeeze)
    #     return mem_sim_squeeze, mem_idx_squeeze


    def SACC(self, sims_forward, pids, memory, threshold):
        # Scene-Aware Cycle Consistency for  "multiple queries" 
        sims_forward[sims_forward<threshold] = 0
        sims_forward_rev = torch.zeros_like(sims_forward)
        if self.use_cycle:
            for  j, (pid, co_sim) in enumerate(zip(pids, sims_forward)):
                matched_sim, matched_idx = self.sacc(co_sim, pid, memory)
                sims_forward_rev[j, matched_idx] = matched_sim
                # print(5)
        else:
            for  j, (pid, co_sim) in enumerate(zip(pids, sims_forward)):
                matched_idx = (co_sim > 0).nonzero().squeeze(1)
                sims_forward_rev[j, matched_idx] = co_sim[matched_idx]
                # print(6)
        return sims_forward_rev

    def predict(self, memory, targets):

        targets_uniq=targets.unique()
        mem_vec = memory[targets_uniq]
        mem_sim = mem_vec.mm(memory.t())

        easy_positive = self.SACC(mem_sim, targets_uniq, memory, self.t)
        easy_positive.scatter_(1, targets_uniq.unsqueeze(1), float(1))

        print('easy_positive')
        if self.use_coap:
            ## CO-APPEARANCE
            for i, target in enumerate(targets_uniq):
                scene_mask = self.cnt2snum[targets_uniq] == self.cnt2snum[target]
                scene_mask[i] = False
                neighbors = scene_mask.nonzero().squeeze(1)
                if len(neighbors) > 0: 
                    ## Compute scene priority
                    advantage = torch.max(easy_positive[neighbors,:],dim=0)[0]
                    penalty = -100*easy_positive[i] # to avoid scene occlusion
                    priority = advantage + penalty
                    ## Expand the co-appearance advantage to scene level
                    scene_priority = torch.zeros((len(self.cnt2snum.unique()),)).cuda().index_add(dim=0, index=self.cnt2snum, source=priority)
                    scene_priority = torch.gather(input=scene_priority, dim=0, index=self.cnt2snum)
                    mem_sim[i] = mem_sim[i] + self.s_c * scene_priority
        print('coapp')
        # Ignore similairties below threshold
        hard_positive = self.SACC(mem_sim, targets_uniq, memory, self.t_c)
        print('hard_positive')
        # Debug
        
        _, cnt = torch.unique(self.cnt2snum[(easy_positive[0] )>0], return_counts=True)
        assert (cnt>1).sum()==0, "easy_positive Scene occlusion"
        _, cnt = torch.unique(self.cnt2snum[( hard_positive[0])>0], return_counts=True)
        assert (cnt>1).sum()==0, "hard_positive Scene occlusion"
        _, cnt = torch.unique(self.cnt2snum[(easy_positive[0] + hard_positive[0])>0], return_counts=True)
        assert (cnt>1).sum()==0, "all_positive Scene occlusion"
        print('predict')
        ## Expand multi-label
        multilabel = (easy_positive + hard_positive)>0
        multilabel_ = torch.zeros(len(targets), mem_sim.shape[1]).bool().cuda()
        for t_uniq, mlabel in zip(targets_uniq, multilabel):
            midx = (t_uniq==targets).nonzero().squeeze(1)
            multilabel_[midx, :] = mlabel.repeat(len(midx), 1)
       
        return multilabel_

    def draw_proposal(self, targets, multilabels):
        # path = './logs/outputs/mAP6/no_t{:.2f}_sc{:.2f}_tc{:.2f}/'.format(self.t, self.s_c, self.t_c)
        # path = './logs/outputs/coap/no_coap/'
        # path = './logs/outputs/coap/no_coap_epoch10/'
        path = './logs/outputs/coap/cycle_th05/'
        # path = './logs/outputs/coap/no_prior_th04/'
        # print('path: ', path)
        flist=glob('./logs/outputs/all/*.jpg')
        for i, (target, multilabel) in enumerate(zip(targets, multilabels)):
            fname=flist[target].split('/')[-1].split('.')[0]
            os.makedirs(path+fname)
            for label in (multilabel==1).nonzero():
                shutil.copy(flist[label], path+fname)
        raise ValueError
