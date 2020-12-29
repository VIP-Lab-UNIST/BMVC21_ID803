import torch

class MPLP(object):
    def __init__(self, t, uniq, k, coap, t_c):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.t = t
        self.uniq=uniq
        self.k = k
        self.coap=coap
        self.t_c = t_c

    def predict(self, memory, scenes, targets, targets_scenes):

        mem_vec = memory[targets]

        # Uniqueness
        s_list=torch.unique(scenes)
        if self.uniq:
            mem_sim_ = mem_vec.mm(memory.t())
            mem_sim=-torch.ones(mem_sim_.shape).to(self.device)
            for s_num in s_list:
                u_topk_val, u_topk_idx=mem_sim_[:,s_num==scenes].topk(k=min(self.k, len((s_num==scenes).nonzero())), dim=1)
                mem_sim[:,s_num==scenes]=mem_sim[:,s_num==scenes].scatter_(1, u_topk_idx, u_topk_val)
        else: mem_sim = mem_vec.mm(memory.t())

        m, n = mem_sim.size()
        mem_simsorted, index_sorted = torch.sort(mem_sim, dim=1, descending=True)
        mask_num = torch.sum(mem_simsorted > self.t, dim=1)

        multilabel = torch.zeros(mem_sim.size()).to(self.device)
        co_cnt = torch.zeros(mem_sim.size()).to(self.device)
        for i, (target, target_scene) in enumerate(zip( targets, targets_scenes)):
            topk = int(mask_num[i].item())
            topk = max(topk, 20)
            topk_idx = index_sorted[i, :topk]
            vec = memory[topk_idx]
            sim = vec.mm(memory.t())

            _, idx_sorted = torch.sort(sim.detach().clone(), dim=1, descending=True)
            step = 1
            for j in range(topk):
                pos = torch.nonzero(idx_sorted[j] == index_sorted[i, 0]).item()
                if pos > topk: break
                step = max(step, j)
            step = step + 1
            step = min(step, mask_num[i].item())
            if step <= 0: continue
            multilabel[i, index_sorted[i, 0:step]] = float(1)

            ## Co-appearance
            if self.coap:
                co_mem_vec = memory[(target_scene==scenes)]
                co_mem_sim = co_mem_vec.mm(memory.t())
                co_cnt[i, :] = torch.sum( (co_mem_sim > self.t_c).type(torch.int), dim=0 ).clamp(max=1)
                co_cnt[i, target_scene==scenes] = 0
                co_cnt[i,index_sorted[i, 0:step]] = 1
            else: co_cnt[i,index_sorted[i, 0:step]] = 1


        targets = torch.unsqueeze(targets, 1)
        multilabel.scatter_(1, targets, float(1))
        for s_num in s_list: co_cnt[:, s_num==scenes]=co_cnt[:, s_num==scenes].sum(dim=1, keepdim=True).clamp(max=len((s_num==scenes).nonzero()))

        return multilabel, co_cnt