import torch

class MPLP(object):
    def __init__(self, total_scene, t, uniq, k, coap, t_c):
        self.total_scene = total_scene
        self.t = t
        self.uniq=uniq
        self.k = k
        self.coap=coap
        self.t_c = t_c

    def predict(self, memory, targets):

        targets_uniq=targets.unique()
        targets_scenes_uniq=self.total_scene[targets_uniq]

        mem_vec = memory[targets_uniq]
        mem_sim = mem_vec.mm(memory.t())

        m, n = mem_sim.size()
        mem_simsorted, index_sorted = torch.sort(mem_sim, dim=1, descending=True)

        multilabel = torch.zeros(mem_sim.size()).cuda()
        co_cnt = torch.zeros(mem_sim.size()).cuda()
        for i, (target, target_scene, simsorted, idxsorted) in enumerate(zip( targets_uniq, targets_scenes_uniq, mem_simsorted, index_sorted)):
            
            ## UNIQUENESS: Select topk, topk_idx
            topk_idx=[]
            scene_sorted=self.total_scene[idxsorted]
            for j, (scn, idx, sim) in enumerate(zip(scene_sorted, idxsorted, simsorted)):
                if scn in scene_sorted[:j]: continue
                topk_idx.append(idx.item())
                if (sim<self.t) & (j>=20): break
            topk=len(topk_idx)

            ## Cycle consistency
            vec = memory[topk_idx]
            sim = vec.mm(memory.t())
            _, idx_sorted = torch.sort(sim.detach().clone(), dim=1, descending=True)
            step = 1
            for j in range(topk):
                pos = torch.nonzero(idx_sorted[j] == index_sorted[i, 0]).item()
                if pos > topk: break
                step = max(step, j)
            step = step + 1
            step = min(step, topk)
            if step <= 0: continue
            multilabel[i, topk_idx[:step]] = float(1)

            ## CO-APPEARANCE: Count the co-appear persons
            co_mem_vec = memory[(target_scene==self.total_scene)]
            co_mem_sim = co_mem_vec.mm(memory.t())
            co_vec = torch.sum( (co_mem_sim > self.t_c).type(torch.int), dim=0 ).clamp(max=1)
            co_vec[target_scene==self.total_scene] = 0
            co_vec[topk_idx[:step]] = 1
            # co_vec[index_sorted[i, topk_idx[:step]]] = 1
            for co_idx in topk_idx[:step]: co_cnt[i, co_idx]=co_vec[self.total_scene[co_idx]==self.total_scene].sum()
            

        # Expand multi-label and co_cnt
        multilabel_=torch.zeros((targets.shape[0], multilabel.shape[1])).cuda()
        co_cnt_=torch.zeros((targets.shape[0], multilabel.shape[1])).cuda()
        for t_uniq, mlabel, ccnt in zip(targets_uniq, multilabel, co_cnt):
            midx = (t_uniq==targets).nonzero()
            multilabel_[midx, :]=mlabel
            co_cnt_[midx, :]=ccnt

        targets = torch.unsqueeze(targets, 1)
        multilabel_.scatter_(1, targets, float(1))

        return multilabel_, co_cnt_





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