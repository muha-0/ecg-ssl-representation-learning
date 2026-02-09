import torch

def info_nce(z, pair_ids, temperature=0.1):
    z = z.float()  # important under autocast
    sim = ((z @ z.T) / temperature).float()

    N = z.size(0)
    mask = torch.eye(N, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, torch.finfo(sim.dtype).min)

    pid = pair_ids
    pos = (pid.unsqueeze(0) == pid.unsqueeze(1)) & (~mask)

    logsumexp = torch.logsumexp(sim, dim=1)
    pos_sim = sim[pos].view(N)
    return -(pos_sim - logsumexp).mean()