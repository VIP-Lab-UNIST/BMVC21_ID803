import torch

class MPLP(object):

    def __init__(self, total_scene, t, t_c, s_c, k):
        self.total_scene = total_scene
        self.t = t
        self.t_c = t_c
        self.s_c = s_c
        self.k = k

    def predict(self, memory, targets):

        targets_uniq=targets.unique()

        ## COAPEARANCE
        co_cnts=torch.zeros(len(targets_uniq), len(memory)).cuda()
        for i, target in enumerate(targets_uniq):
            co_vec = memory[self.total_scene==self.total_scene[target]]
            co_sim = co_vec.mm(memory.t())
            co_cnts[i, :]=torch.max(co_sim, dim=0)[0]
            co_cnts[i, :][co_cnts[i, :] < self.t_c] = 0
            co_cnts[i, :] *= self.s_c
            co_cnts[i, self.total_scene==self.total_scene[target]] = 0
            co_cnts[i, target] = 1

        for scn_num in self.total_scene.unique(): co_cnts[:, self.total_scene==scn_num]=co_cnts[:, self.total_scene==scn_num].sum(dim=1).unsqueeze(1)

        mem_vec = memory[targets_uniq]
        mem_sim = mem_vec.mm(memory.t())
        mem_sim = (mem_sim + co_cnts).clamp(max=1.)

        m, n = mem_sim.size()
        mem_simsorted, index_sorted = torch.sort(mem_sim, dim=1, descending=True)
        multilabel = torch.zeros(mem_sim.shape).cuda()
        for i, (target, simsorted, idxsorted) in enumerate(zip( targets_uniq, mem_simsorted, index_sorted)):
            
            ## UNIQUENESS: Select candidate(top k)
            topk_sim=[]
            topk_idx=[]
            topk_scn=[]
            for sim, idx in zip(simsorted, idxsorted):
                if self.total_scene[idx] in topk_scn: continue
                if (sim < self.t): break
                topk_sim.append(sim)
                topk_scn.append(self.total_scene[idx])
                topk_idx.append(idx)
                
            topk_sim = torch.tensor(topk_sim).cuda()
            topk_idx = torch.tensor(topk_idx).cuda()
            num_topk = len(topk_scn)
            assert len(self.total_scene[topk_idx]) == len(self.total_scene[topk_idx].unique())

            ### NO UNIQUENESS: Cycle consistency
            topk_vec = memory[topk_idx]
            topk_sim = topk_vec.mm(memory.t())
            topk_sim_sorted, topk_idx_sorted = torch.sort(topk_sim.detach().clone(), dim=1, descending=True)

            cycle_idx = []
            for j in range(num_topk):
                pos = torch.nonzero(topk_idx_sorted[j] == target).item()
                if pos > max(num_topk, self.k): continue
                cycle_idx.append(topk_idx_sorted[j, 0])
            cycle_idx = torch.tensor(cycle_idx).cuda()

            if len(cycle_idx) == 0: multilabel[i, target] = float(1)
            else: multilabel[i, cycle_idx] = float(1)

        # Expand multi-label
        multilabel_=torch.zeros((targets.shape[0], multilabel.shape[1])).cuda()
        for t_uniq, mlabel in zip(targets_uniq, multilabel):
            midx = (t_uniq==targets).nonzero()
            multilabel_[midx, :]=mlabel
        
        targets = torch.unsqueeze(targets, 1)
        multilabel_.scatter_(1, targets, float(1))
        assert 0 not in multilabel_.sum(dim=1)

        return multilabel_





        ## UNIQUENESS: Cycle consistency
        # cmem_vec = memory[topk_idx]
        # cmem_sim = cmem_vec.mm(memory.t())

        # multilabel_idx=[]
        # cmem_simsorted, cindex_sorted = torch.sort(cmem_sim.detach().clone(), dim=1, descending=True)
        # for j, (t_idx, csim_sorted, cidxsorted) in enumerate(zip(topk_idx, cmem_simsorted, cindex_sorted)):
        #     ctopk_idx=[]
        #     cscene_sorted=self.total_scene[cidxsorted]
        #     for k, (cscn, csim, cidx) in enumerate(zip(cscene_sorted, csim_sorted, cidxsorted)):
        #         if cscn in cscene_sorted[:k]: continue
        #         ctopk_idx.append(cidx.item())
        #         if len(ctopk_idx)==topk: break

        #     if ctopk_idx in idxsorted[0]: multilabel_idx.append(t_idx)
        #     else: break
        # multilabel[i, multilabel_idx] = float(1)