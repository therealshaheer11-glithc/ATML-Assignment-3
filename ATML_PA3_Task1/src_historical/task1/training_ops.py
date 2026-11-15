"""Same response-token CE objective, with checkpointed head chunks to bound logits memory."""
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def chunked_response_sum(model, ids, labels, attention_mask, chunk_size=128):
    output=model.model(input_ids=ids,attention_mask=attention_mask,use_cache=False,return_dict=True)
    hidden=output.last_hidden_state[:,:-1,:].reshape(-1,output.last_hidden_state.shape[-1])
    targets=labels[:,1:].reshape(-1)
    valid=targets.ne(-100);hidden=hidden[valid];targets=targets[valid]
    count=targets.numel()
    if not count:raise ValueError('Effective batch contains no supervised tokens')
    total=torch.zeros((),device=hidden.device,dtype=torch.float32)
    def head_loss(h,t):
        return F.cross_entropy(model.lm_head(h).float(),t,reduction='sum')
    for start in range(0,count,chunk_size):
        total=total+checkpoint(head_loss,hidden[start:start+chunk_size],targets[start:start+chunk_size],
                               use_reentrant=False,preserve_rng_state=False)
    return total,count


def check_chunked_gradients():
    from torch import nn
    from types import SimpleNamespace
    class TinyBase(nn.Embedding):
        def forward(self,input_ids,**kwargs):return SimpleNamespace(last_hidden_state=super().forward(input_ids))
    torch.manual_seed(6304)
    model=nn.Module();model.model=TinyBase(17,7);model.lm_head=nn.Linear(7,17,bias=False)
    ids=torch.randint(0,17,(2,9));labels=ids.clone();labels[0,:3]=-100;labels[1,:6]=-100
    loss,count=chunked_response_sum(model,ids,labels,torch.ones_like(ids),chunk_size=3)
    first=torch.autograd.grad(loss/count,list(model.parameters()))
    logits=model.lm_head(model.model(ids).last_hidden_state)
    reference=F.cross_entropy(logits[:,:-1].reshape(-1,17),labels[:,1:].reshape(-1),ignore_index=-100,reduction='sum')/count
    second=torch.autograd.grad(reference,list(model.parameters()))
    torch.testing.assert_close(loss/count,reference,rtol=1e-4,atol=1e-5)
    for a,b in zip(first,second):torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-5)
    return {'chunked_response_loss_matches_dense':True,'all_parameter_gradients_match_dense':True,'chunk_tokens':128}
