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

    def SACC(self, sim_forward, memory, query_pid):
        # Scene-Aware Cycle Consistency
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

    def predict(self, memory, targets):

        targets_uniq=targets.unique()
        mem_vec = memory[targets_uniq]
        mem_sim = mem_vec.mm(memory.t())
        multilabel = torch.zeros(mem_sim.shape).cuda()
        
        for i, (target, sim) in enumerate(zip( targets_uniq, mem_sim)):
            # print(1)
            ## COAPEARANCE
            if self.use_coap:
                # Calculate co-appearance similarity
                persons_in_query_image = (self.cnt2snum==self.cnt2snum[target])
                persons_in_query_image[target] = False
                neighbors = persons_in_query_image.nonzero().squeeze(1)
                # print(2)
                if len(neighbors) > 0: 
                    ## Compute similairties of co-persons to the other objects
                    co_vec = memory[neighbors]
                    co_persons = torch.cat([neighbors.view(-1), target.view(-1)], dim=0)
                    co_persons, _ = torch.sort(co_persons, dim=0, descending=False)
                    co_sims_forward = co_vec.mm(memory.t())
                    # print(3)

                    ## Remove similarities of query and co-persons itself
                    ## and ignore similarities lower than a threshold
                    co_sims_forward[:, co_persons] = 0
                    co_sims_forward[co_sims_forward < self.t_c] = 0
                    # print(4)
                    ## Apply Scene-aware cycle consistency (SACC)
                    advantage = torch.zeros_like(co_sims_forward)
                    if self.use_cycle:
                        for  j, (pid, co_sim) in enumerate(zip(neighbors, co_sims_forward)):
                            matched_sim, matched_idx = self.SACC(co_sim, memory, pid)
                            advantage[j, matched_idx] = matched_sim
                            # print(5)
                    else:
                        for  j, (pid, co_sim) in enumerate(zip(neighbors, co_sims_forward)):
                            matched_idx = (co_sim > 0).nonzero().squeeze(1)
                            advantage[j, matched_idx] = co_sim[matched_idx]
                            # print(6)
                    advantage = torch.max(advantage, dim=0)[0]
                    # print(7)
                    ## Expand the co-appearance advantage to scene level
                    # Debuging: 
                    assert max(advantage)<=1, "Debug: mlplp.py: advantage is lager than 1"
                    co_sim_sum = torch.zeros((len(self.cnt2snum.unique()), )).cuda().index_add(dim=0, index=self.cnt2snum, source=advantage)
                    # print(8)
                    # Debuging: 
                    assert max(co_sim_sum.view(-1))<=1, "Debug: mlplp.py: co_sim_sum is lager than 1"
                    co_sim_sum = torch.gather(input=co_sim_sum, dim=0, index=self.cnt2snum)
                    # print(9)
                    # Debuging: 
                    assert max(co_sim_sum.view(-1))<=1, "Debug: mlplp.py: co_sim_sum is lager than 1"
                    sim = sim + self.s_c * co_sim_sum
                    # print(10)
            # Ignore similairties below threshold
            sim[sim < self.t] = 0

            ## Apply Scene-aware cycle consistency (SACC)
            if self.use_cycle:
                _, matched_idx = self.SACC(sim, memory, target)
                # print(11)
                multilabel[i, matched_idx] = float(1)
                multilabel[i, target] = float(1)
                # print(12)
            else: 
                matched_idx = (sim > 0).nonzero().squeeze(1)
                multilabel[i, matched_idx] = float(1)
                multilabel[i, target] = float(1)

        ## Expand multi-label
        multilabel_= torch.zeros((targets.shape[0], multilabel.shape[1])).cuda()
        for t_uniq, mlabel in zip(targets_uniq, multilabel):
            midx = (t_uniq==targets).nonzero().squeeze(1)
            multilabel_[midx, :] = mlabel.repeat(len(midx), 1)
        # print(13)
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
