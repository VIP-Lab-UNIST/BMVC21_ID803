import torch

class MPLP(object):
    def __init__(self, t=0.6, k=1):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.t = t
        self.k = k

    def predict(self, memory, scenes, targets, targets_scenes):

        mem_vec = memory[targets]
        mem_sim = mem_vec.mm(memory.t())

        # Uniqueness
        s_list=torch.unique(scenes)
        mem_sim=-torch.ones(mem_sim_.shape).cuda()
        for s_num in s_list:
            u_topk_val, u_topk_idx=mem_sim_[:,s_num==scenes].topk(k=min(self.k, len((s_num==scenes).nonzero())), dim=1)
            mem_sim[:,s_num==scenes]=mem_sim[:,s_num==scenes].scatter_(1, u_topk_idx, u_topk_val)

        m, n = mem_sim.size()
        mem_simsorted, index_sorted = torch.sort(mem_sim, dim=1, descending=True)
        mask_num = torch.sum(mem_simsorted > self.t, dim=1)

        multilabel = torch.zeros(mem_sim.size()).to(self.device)
        for i in range(m):
            topk = int(mask_num[i].item())
            topk = max(topk, 10)
            topk_idx = index_sorted[i, :topk]
            vec = memory[topk_idx]
            sim = vec.mm(memory.t())

            # Uniqueness for cycle-consistency
            # sim=-torch.ones(sim_.shape).cuda()
            # for s_num in s_list:
            #     u_topk_val, u_topk_idx=sim_[:,s_num==scenes].topk(k=min(self.k, len((s_num==scenes).nonzero())), dim=1)
            #     sim[:,s_num==scenes]=sim[:,s_num==scenes].scatter_(1, u_topk_idx, u_topk_val)

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

        # (N, 18048)   
        targets = torch.unsqueeze(targets, 1)
        multilabel.scatter_(1, targets, float(1))
        
        co_appearance_cnt=0
        return multilabel, co_appearance_cnt