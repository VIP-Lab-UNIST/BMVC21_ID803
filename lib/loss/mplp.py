import torch
import shutil 
import os
from glob import glob

class MPLP(object):

    def __init__(self, total_scene, t, t_c, s_c, k):
        self.cnt2snum = total_scene
        self.t = t
        self.t_c = t_c
        self.s_c = s_c
        self.k = k

    def predict(self, memory, targets):

        targets_uniq=targets.unique()
        mem_vec = memory[targets_uniq]
        mem_sim = mem_vec.mm(memory.t())
        
        multilabel = torch.zeros(mem_sim.shape).cuda()
        for i, (target, sim) in enumerate(zip( targets_uniq, mem_sim)):

            ## COAPEARANCE
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
            topk_sim=[]
            topk_idx=[]
            topk_scn=[]
            for j, (sim, idx) in enumerate(zip(simsorted, idxsorted)):
                if self.cnt2snum[idx] in topk_scn: continue
                if (sim < self.t): break
                topk_sim.append(sim)
                topk_scn.append(self.cnt2snum[idx])
                topk_idx.append(idx)
            topk_sim = torch.tensor(topk_sim).cuda()
            topk_idx = torch.tensor(topk_idx).cuda()
            num_topk = len(topk_scn)

            ## NO UNIQUENESS: Cycle consistency
            topk_vec = memory[topk_idx]
            topk_sim = topk_vec.mm(memory.t())
            topk_sim_sorted, topk_idx_sorted = torch.sort(topk_sim.detach().clone(), dim=1, descending=True)

            cycle_idx = []
            for j in range(num_topk):
                pos = torch.nonzero(topk_idx_sorted[j] == target).item()
                # if pos > max(num_topk, self.k): break
                if pos > max(num_topk, self.k): continue
                cycle_idx.append(topk_idx_sorted[j, 0])
            
            if len(cycle_idx) == 0: multilabel[i, target] = float(1)
            else: 
                cycle_idx = torch.tensor(cycle_idx).cuda()    
                multilabel[i, cycle_idx] = float(1)

        ## Expand multi-label
        multilabel_=torch.zeros((targets.shape[0], multilabel.shape[1])).cuda()
        for t_uniq, mlabel in zip(targets_uniq, multilabel):
            midx = (t_uniq==targets).nonzero()
            multilabel_[midx, :]=mlabel
        targets = torch.unsqueeze(targets, 1)
        multilabel_.scatter_(1, targets, float(1))

        # self.draw_proposal(targets_uniq, multilabel)

        return multilabel_


    def draw_proposal(self, targets, multilabels):
        path = './logs/outputs/sc{:.2f}_th{:.2f}__/'.format(self.s_c, self.t_c)
        print('path: ', path)
        flist=glob('./logs/outputs/all/*.jpg')
        for i, (target, multilabel) in enumerate(zip(targets, multilabels)):
            fname=flist[target].split('/')[-1].split('.')[0]
            os.makedirs(path+fname)
            for label in multilabel.nonzero():
                shutil.copy(flist[label], path+fname)
        raise ValueError
