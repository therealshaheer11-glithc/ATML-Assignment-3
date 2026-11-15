"""Shared mathematical operations for the five prescribed Task 1 methods."""
import math
import torch
from torch import nn
from torch.nn import functional as F


TARGETS = ('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj')


def response_loss_sum(logits, labels):
    targets = labels[:, 1:].reshape(-1)
    total = F.cross_entropy(logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
                           targets, ignore_index=-100, reduction='sum')
    return total, int(targets.ne(-100).sum())


def learning_rate(update, peak):
    if not 1 <= update <= 1250:
        raise ValueError('Update outside the prescribed budget')
    if update <= 38:
        return peak * update / 38
    return peak * .5 * (1 + math.cos(math.pi*(update-38)/(1250-38)))


def l2_sp(model, reference):
    parameters = dict(model.named_parameters())  # Removes tied-parameter duplicates.
    if parameters.keys() != reference.keys():
        raise ValueError('Reference/model unique parameter names differ')
    return sum(.5*(p-reference[name]).square().sum() for name,p in parameters.items())


def projected_factors(A, B, U=None, V=None):
    if (U is None) != (V is None):
        raise ValueError('OPLoRA requires both protected bases')
    if U is None:
        return A, B
    if any(x.dtype != torch.float32 for x in (A,B,U,V)):
        raise TypeError('Projection factors and bases must remain FP32')
    with torch.autocast(device_type=A.device.type, enabled=False):
        return A-(A@V)@V.T, B-U@(U.T@B)


class AdapterLinear(nn.Module):
    def __init__(self, base, A, B, U=None, V=None, alpha=32, dropout=.05):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.A = nn.Parameter(A.detach().float().clone())
        self.B = nn.Parameter(B.detach().float().clone())
        self.register_buffer('U', None if U is None else U.detach().float().clone())
        self.register_buffer('V', None if V is None else V.detach().float().clone())
        self.scale = alpha/A.shape[0]
        self.dropout = nn.Dropout(dropout)

    def effective_factors(self):
        return projected_factors(self.A,self.B,self.U,self.V)

    def forward(self,x):
        A,B = self.effective_factors()
        return self.base(x)+self.scale*F.linear(F.linear(self.dropout(x),A.to(x.dtype)),B.to(x.dtype))


def cpu_math_checks():
    torch.manual_seed(6304)
    # Compare accumulated token-normalized gradients with a direct per-token sum.
    x = torch.randn(3,7,5)
    labels = torch.randint(0,11,(3,7)); labels[0,:4]=-100;labels[1,:2]=-100;labels[2,:5]=-100
    weight = torch.randn(5,11,requires_grad=True)
    denominator = int(labels[:,1:].ne(-100).sum())
    accumulated = sum(response_loss_sum(x[i:i+1]@weight,labels[i:i+1])[0]/denominator for i in range(3))
    grad = torch.autograd.grad(accumulated,weight)[0]
    logits = x@weight
    individual = [F.cross_entropy(logits[i,j-1:j],labels[i,j:j+1])
                  for i in range(3) for j in range(1,7) if labels[i,j] != -100]
    expected = torch.autograd.grad(torch.stack(individual).mean(),weight)[0]
    torch.testing.assert_close(grad,expected,rtol=1e-4,atol=1e-5)
    # Count a deliberately tied parameter once, while including a bias parameter.
    model = nn.Module(); model.register_parameter('p',nn.Parameter(torch.tensor([1.,2.])))
    model.register_parameter('alias',model.p); model.register_parameter('bias',nn.Parameter(torch.tensor([3.])))
    reference = {name:torch.zeros_like(p) for name,p in model.named_parameters()}
    penalty = l2_sp(model,reference)
    torch.testing.assert_close(penalty,torch.tensor(7.))
    penalty.backward();torch.testing.assert_close(model.p.grad,model.p.detach())
    torch.testing.assert_close(model.bias.grad,model.bias.detach())
    # Compare factor projection against explicit dense projectors on a tiny fixture.
    W = torch.randn(24,32); U,S,Vh = torch.linalg.svd(W,full_matrices=False)
    U=U[:,:4].contiguous(); V=Vh[:4].T.contiguous()
    A=torch.empty(4,32);nn.init.kaiming_uniform_(A,a=math.sqrt(5));B=torch.randn(24,4)*.01
    ae,be=projected_factors(A,B,U,V)
    delta=2*be@ae; dense=2*(torch.eye(24)-U@U.T)@B@A@(torch.eye(32)-V@V.T)
    torch.testing.assert_close(delta,dense,rtol=1e-4,atol=1e-5)
    assert float(torch.linalg.norm(U.T@delta)/torch.linalg.norm(delta))<=1e-5
    assert float(torch.linalg.norm(delta@V)/torch.linalg.norm(delta))<=1e-5
    layer=AdapterLinear(nn.Linear(32,24,bias=False),A,B,U,V,alpha=8,dropout=0)
    with torch.no_grad():layer.base.weight.copy_(W)
    inp=torch.randn(2,32); actual=layer(inp)
    torch.testing.assert_close(actual,F.linear(inp,W+delta),rtol=1e-4,atol=1e-5)
    actual.square().mean().backward()
    assert layer.base.weight.grad is None and layer.A.grad is not None and layer.B.grad is not None
    assert all(torch.isfinite(p.grad).all() for p in (layer.A,layer.B))
    assert learning_rate(38,1)==1 and learning_rate(1250,1)==0
    return {'token_weighted_gradient':True,'l2_sp_unique_parameters_and_biases':True,
            'double_projection_and_leakage':True,'factor_dense_and_merge_equivalence':True,
            'frozen_backbone_and_adapter_gradients':True,'learning_rate_endpoints':True}
