"""
MoE Models which differ only in the router variant.

They all have the same SwiGLU MLP and DeepSeek-style auxiliary-loss free load balancing mechanism.

Note: In general, we initialize parameters normally with standard deviation 1/sqrt(D)
"""



import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import math
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
if device.type == "cuda":
    from momoe import MoMoE

"""
Dimension Key:

B: batch size
S: sequence length
M: B * S
D: embedding dimension
H: hidden dimension in expert networks
N: number of experts
K: number of chosen experts per token
"""


"""
SwiGLU MLP
"""
class SwiGLUMLP(nn.Module):
    def __init__(self, D, H):
        super().__init__()
        self.D = D
        self.H = H
        self.lin1 = nn.Linear(D, 2*H)
        self.lin2 = nn.Linear(H, D)
        self.lin1.weight.data.normal_(0, 1/math.sqrt(D))
        self.lin1.bias.data.zero_()
        self.lin2.weight.data.normal_(0, 1/math.sqrt(H))
        self.lin2.bias.data.zero_()

        
    def forward(self, x_BSD):
        a_BSH, b_BSH = self.lin1(x_BSD).chunk(2, dim=-1)
        y_BSD = self.lin2(F.silu(a_BSH) * b_BSH)
        return y_BSD


"""
Router Variants
"""
class RegularRouter(nn.Module):
    def __init__(self, D, N):
        super().__init__()
        self.gate = nn.Linear(D, N)
        self.gate.weight.data.normal_(0, 1/math.sqrt(D))
        self.gate.bias.data.zero_()
    def forward(self, x_BSD): 
        return self.gate(x_BSD)

class RandomRouter(nn.Module):
    def __init__(self, D, N):
        super().__init__()
        self.gate = nn.Linear(D, N)
        self.gate.weight.data.normal_(0, 1/math.sqrt(D))
        self.gate.bias.data.zero_()
        self.gate.requires_grad_(False)  # freeze weights, grads still flow to input
    def forward(self, x_BSD):
        return self.gate(x_BSD)

class OrthogonalRouter(nn.Module):
    def __init__(self, D, N):
        super().__init__()
        # random semi-orthogonal matrix
        W_DN = torch.randn(D, N) / math.sqrt(D)
        W_DN = torch.linalg.qr(W_DN)[0][:, :N]
        self.register_buffer('W_DN', W_DN)
        self.register_buffer('b_N', torch.zeros(N))
    def forward(self, x_BSD):
        return x_BSD @ self.W_DN + self.b_N[None, None, :]

class HashRouter(nn.Module):
    def __init__(self, D, N):
        super().__init__()
        # random unit-norm hyperplanes
        W_DN = torch.randn(D, N)
        W_DN = W_DN / W_DN.norm(dim=0, keepdim=True)
        self.register_buffer('W_DN', W_DN)
    def forward(self, x_BSD):
        return x_BSD @ self.W_DN


"""
MoE Base Class
"""
class MoE(nn.Module):
    def __init__(self, D, H, N, K, router, bias_rate=0.001):
        super().__init__()
        self.D = D
        self.H = H
        self.N = N
        self.K = K
        self.bias_rate = bias_rate
        assert self.K <= self.N, "K must be less than or equal to N"

        if device.type == "1234":
            self.momoe = MoMoE(
                embedding_dim=D, 
                intermediate_dim=H,
                num_experts=N,
                num_chosen_experts=K,
                save_percent=100,
                Wl1_ND2H=None,
                Wl2_NHD=None,
            )
        else:
            self.experts = nn.ModuleList([SwiGLUMLP(D, H) for _ in range(N)])
        self.router = router
        self.register_buffer("biases_N", torch.zeros(N))

    def forward(self, x_BSD):
        B = x_BSD.size(0)
        S = x_BSD.size(1)
        
        # apply router to get weights for each expert
        scores_BSN = self.router(x_BSD)

        # get top K experts for each token
        w_BSN = F.softmax(scores_BSN, dim=-1)
        _, idx_BSK = torch.topk(scores_BSN + self.biases_N[None, None, :], self.K, dim=-1)
        val_BSK = w_BSN.gather(-1, idx_BSK)
        val_BSK = val_BSK / val_BSK.sum(dim=-1, keepdim=True)

        # flatten for simplicity
        x_MD = einops.rearrange(x_BSD, "B S D -> (B S) D")
        val_MK = einops.rearrange(val_BSK, "B S K -> (B S) K")
        idx_MK = einops.rearrange(idx_BSK, "B S K -> (B S) K")



        # naive version, apply all experts to all tokens
        if device.type != "cuda" or device.type == "cuda":
            expert_outputs_NMD = torch.stack([expert(x_MD) for expert in self.experts], dim=0)
            expert_weights_MN = torch.zeros(x_MD.size(0), self.N, device=x_MD.device)
            expert_weights_MN.scatter_(1, idx_MK, val_MK)
            y_MD = torch.einsum('mn,nmd->md', expert_weights_MN, expert_outputs_NMD)
            counts_N = torch.bincount(idx_MK.flatten(), minlength=self.N).float()
        else:
            s_NM = einops.rearrange(scores_BSN, "B S N -> N (B S)")
            mask_MN = torch.zeros(x_MD.size(0), self.N, device=x_MD.device, dtype=torch.int32)
            mask_MN.scatter_(1, idx_MK, 1)
            mask_NM = mask_MN.T
            y_BSD, counts_N = self.momoe(x_BSD, mask_NM, s_NM)
            y_MD = einops.rearrange(y_BSD, "B S D -> (B S) D")




        # # sparse version, works faster in theory but slower on mps with small params
        # y_MD = torch.zeros_like(x_MD)
        # for i, expert in enumerate(self.experts):
        #     mask_MK = (idx_MK == i)
        #     tok_idx, k_idx = torch.where(mask_MK)
        #     out = expert(x_MD[tok_idx])
        #     y_MD.index_add_(0, tok_idx, val_MK[tok_idx, k_idx][:, None] * out)


        # update biases
        if self.training:
            with torch.no_grad():
                self.biases_N -= self.bias_rate * (counts_N - counts_N.float().mean()).sign()

        return einops.rearrange(y_MD, "(B S) D -> B S D", B=B, S=S)


"""
MoE Variants
"""
class RegularMoE(MoE):
    def __init__(self, D, H, N, K):
        super().__init__(D, H, N, K, RegularRouter(D, N))

class RandomMoE(MoE):
    def __init__(self, D, H, N, K):
        super().__init__(D, H, N, K, RandomRouter(D, N))

class OrthogonalMoE(MoE):
    def __init__(self, D, H, N, K):
        super().__init__(D, H, N, K, OrthogonalRouter(D, N))

class HashMoE(MoE):
    def __init__(self, D, H, N, K):
        super().__init__(D, H, N, K, HashRouter(D, N))