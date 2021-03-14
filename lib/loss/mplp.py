import torch
import shutil 
import os
import numpy as np
import json

from tqdm import tqdm
from functools import reduce
from glob import glob

class MPLP(object):

    def __init__(self, use_hnm, use_hpm, total_scene, threshold, coapp_scale):
        self.cnt2snum = total_scene
        with open('./dist_dict.json', 'r') as fp:
            self.dist_mat = json.load(fp)
        self.threshold = threshold
        self.coapp_scale = coapp_scale
        self.use_hnm = use_hnm
        self.use_hpm = use_hpm
        
    def forward_matching_1d(self, sim_forward):

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
        
        return cand_idx 

    def backward_matching_1d(self, candidates_pid, query_pid, co_app, memory):
        ## Compute local person index
        co_persons = (self.cnt2snum==self.cnt2snum[query_pid]).nonzero().squeeze(1)
        co_persons, _ = torch.sort(co_persons, dim=0, descending=False)
        offset = co_persons[0]
        
        ## Backward matching
        if len(candidates_pid) > 0:
            candidates_feats = memory[candidates_pid] 
            target_feats = memory[co_persons]
            sims = candidates_feats.mm(target_feats.t()) 
            amp_sims = sims + co_app[candidates_pid].unsqueeze(1)
            backward_matched_idx = amp_sims.max(dim=1)[1]
            cycle_matched_idx = candidates_pid[backward_matched_idx==(query_pid-offset)]
            return cycle_matched_idx
        else:
            # empty index tensor
            return candidates_pid 

    def cycle_consistency_1d(self, sim, coapp, qid, memory):
        forward_matched_idx  = self.forward_matching_1d(sim+coapp)
        cycle_matched_idx    = self.backward_matching_1d(forward_matched_idx, qid, coapp, memory)
        return cycle_matched_idx

    def hard_negative_mining(self, sims, coapp, pids, memory, threshold):
        
        ## Thresholding
        amp_sims = sims + coapp
        coapp[amp_sims<threshold] = 0
        sims[amp_sims<threshold] = 0
        
        ## Cyclic checking: forward and backward checking
        sims_rev = torch.zeros_like(sims)
        for  j, (pid, sim, co) in enumerate(zip(pids, sims, coapp)):
            valid_matched_idx = self.cycle_consistency_1d(sim, co, pid, memory)
            sims_rev[j, valid_matched_idx] = sim[valid_matched_idx]
                
        return sims_rev

    def co_appearance_factor(self, mem_sim, targets_uniq, positive_mask):
        coapp_factor = []
        current_positives = positive_mask * mem_sim
        for i, target in enumerate(targets_uniq):
            image_pids = (self.cnt2snum[targets_uniq] == self.cnt2snum[target]).nonzero().squeeze(1)
            
            ## (Practical) Give panalty to avoid revisiting a scene having a person matching to a query
            priority = -1000*positive_mask[i] 
            if len(image_pids) > 0: 
                # Squeeze rows
                advantage = torch.max(current_positives[image_pids,:], dim=0)[0] 
                priority += advantage 
        
            ## Expand the priority to scene level
            scene_priority = torch.zeros((len(self.cnt2snum.unique()),)).cuda().index_add(dim=0, index=self.cnt2snum, source=priority)
            scene_priority = torch.gather(input=scene_priority, dim=0, index=self.cnt2snum)
            coapp_factor.append(scene_priority)

        coapp_factor = self.coapp_scale * torch.stack(coapp_factor)

        return coapp_factor

    def hard_positive_mining(self, mem_sim, targets_uniq, positive_mask, memory):

        ## Co-appearance factor
        coapp_factor = self.co_appearance_factor(mem_sim, targets_uniq, positive_mask)

        if self.use_hnm:
            ## Hard negative mining
            hard_positive = self.hard_negative_mining(mem_sim, coapp_factor, targets_uniq, memory, self.threshold)
        else:
            ## Naive thresholding
            hard_positive = mem_sim.clone()
            hard_positive[(mem_sim + coapp_factor) < self.threshold] = 0
            
        return hard_positive 

    def predict(self, memory, targets):

        targets_uniq=targets.unique()
        mem_vec = memory[targets_uniq]
        mem_sim = mem_vec.mm(memory.t())
        
        if self.use_hnm:
            coapp = torch.zeros_like(mem_sim)
            easy_positive = self.hard_negative_mining(mem_sim.clone(), coapp, targets_uniq, memory, self.threshold)
        else:
            easy_positive = mem_sim.clone()
            easy_positive[mem_sim < self.threshold] = 0

        easy_positive.scatter_(1, targets_uniq.unsqueeze(1), float(1))
        
        ## Predict hard positive samples
        multilabel = (easy_positive > 0).float()
        if self.use_hpm:
            ## Iterative hard positive mining 
            for p in range(3):
                hard_positive = self.hard_positive_mining(mem_sim.clone(), targets_uniq, multilabel, memory)
                hard = hard_positive > 0 
                multilabel[hard] = 1
                if (hard.sum()==0):
                    break
        
        ## Expanding multilabels
        multilabel_ = torch.zeros(len(targets), mem_sim.shape[1]).cuda()
        for t_uniq, mlabel in zip(targets_uniq, multilabel):
            midx = (t_uniq==targets).nonzero().squeeze(1)
            multilabel_[midx, :] = mlabel.repeat(len(midx), 1)
            
        return multilabel_


