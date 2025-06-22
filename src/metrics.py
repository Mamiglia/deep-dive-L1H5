import torch
from jaxtyping import Int
from torch import Tensor

@torch.no_grad()
def compute_rank(attn: Tensor ) -> tuple[Tensor,Tensor]:
    attn_cpu=attn.to('cpu')
    sorted_idx = torch.argsort(attn_cpu, dim=-1, descending=True).to(attn.device)
    n=attn.shape[0]
    diag = torch.arange(n, device=attn.device)
    rank = (sorted_idx == diag.unsqueeze(1)).nonzero()[:,1]
    return rank, sorted_idx

def mrr(rank: Tensor) -> float:
    return (1/(rank+1)).mean().item()

def accuracy_k(rank: Tensor, k=32) -> float:
    return (rank<k).float().mean().item()

def precision_k(pred: Tensor, gt: Tensor, k=128) -> float:
    n=pred.size(0)
    return (((pred<k)&(gt<k)).sum()/k/n).item()

def spearman(pred: Tensor, gt: Tensor) -> float:
    n=pred.size(-1)
    idx = torch.arange(n, device=pred.device)
    pr = torch.zeros_like(pred); gr=torch.zeros_like(gt)
    pr.scatter_(1,pred,idx); gr.scatter_(1,gt,idx)
    d2=(pr-gr).pow(2).sum(-1)
    rho = 1 - 6*d2/(n*(n*n-1))
    return rho.mean().item()

def explained_attn_score(gt, pred) -> Tensor:
    if gt.ndim==4 and pred.ndim==3:
        pred=pred.unsqueeze(-3)
    prob=(gt*pred).sum(-1)
    return -prob.clamp(min=1e-12).log().mean(-1)
