"""Successful-update transactions and full mutable-state checkpointing.

Original pretrained tensors are identified by the immutable source archive.
Only connector/adapters and all mutable training state are serialized here.
"""
import copy
import hashlib
import io
import random
from pathlib import Path
import numpy as np
import torch
from .modeling import is_lora,set_stage,assert_precision,trainable_parameters,training_loss
from .protocol import STAGES,learning_rate
from .session_io import bounded_io,save_bytes


def cpu_copy(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {key:cpu_copy(row) for key,row in value.items()}
    if isinstance(value,list):return [cpu_copy(row) for row in value]
    if isinstance(value,tuple):return tuple(cpu_copy(row) for row in value)
    return copy.deepcopy(value)


def assert_equal(left,right,path='state'):
    if isinstance(left,torch.Tensor):
        if not isinstance(right,torch.Tensor) or left.dtype!=right.dtype or left.shape!=right.shape or not torch.equal(left.cpu(),right.cpu()):
            raise AssertionError('Checkpoint tensor differs: '+path)
    elif isinstance(left,np.ndarray):
        if not isinstance(right,np.ndarray) or left.dtype!=right.dtype or not np.array_equal(left,right):
            raise AssertionError('Checkpoint array differs: '+path)
    elif isinstance(left,dict):
        if not isinstance(right,dict) or left.keys()!=right.keys():raise AssertionError('Checkpoint keys differ: '+path)
        for key in left:assert_equal(left[key],right[key],path+'/'+str(key))
    elif isinstance(left,(list,tuple)):
        if type(left)!=type(right) or len(left)!=len(right):raise AssertionError('Checkpoint sequence differs: '+path)
        for index,(a,b) in enumerate(zip(left,right)):assert_equal(a,b,path+'/'+str(index))
    elif type(left)!=type(right) or left!=right:
        raise AssertionError('Checkpoint value differs: '+path)


def capture_rng(device):
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch_cpu':torch.get_rng_state(),
        'torch_cuda':torch.cuda.get_rng_state_all() if device.type=='cuda' else []}


def restore_rng(state,device):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'].cpu())
    if device.type=='cuda':torch.cuda.set_rng_state_all([value.cpu() for value in state['torch_cuda']])


def rng_witness(device):
    """Validation-only random draws; never called by the production update."""
    return {'python':random.random(),'numpy':float(np.random.random()),
        'torch_cpu':torch.rand(4),'torch_device':torch.rand(4,device=device).cpu()}


def save_checkpoint(folder,name,state):
    buffer=io.BytesIO()
    torch.save(state,buffer)
    content=buffer.getvalue()
    sha=hashlib.sha256(content).hexdigest()
    path=Path(folder)/(name+'_'+sha[:16]+'.pt')
    with bounded_io(120,'Full checkpoint save and readback'):
        save_bytes(path,content)
        observed=path.read_bytes()
        if hashlib.sha256(observed).hexdigest()!=sha:raise IOError('Checkpoint readback checksum differs.')
        restored=torch.load(io.BytesIO(observed),map_location='cpu',weights_only=False)
        assert_equal(state,restored)
    return {'path':str(path),'sha256':sha,'bytes':len(content),'readback_verified':True}


def load_checkpoint(receipt):
    with bounded_io(120,'Trusted own checkpoint readback'):
        content=Path(receipt['path']).read_bytes()
        if len(content)!=receipt['bytes'] or hashlib.sha256(content).hexdigest()!=receipt['sha256']:
            raise IOError('Checkpoint does not match its verified receipt.')
    # This is our own content-hashed training state, including NumPy/Python RNG.
    return torch.load(io.BytesIO(content),map_location='cpu',weights_only=False)


class TrainingSession:
    def __init__(self,model,connector,sampler_factory,tables,get_features,identity,device,initial_scale=65536.0):
        self.model,self.connector=model,connector
        self.sampler_factory,self.tables,self.get_features=sampler_factory,tables,get_features
        self.identity=copy.deepcopy(identity)
        self.device=torch.device(device)
        self.initial_scale=initial_scale
        self.stage=1
        self.stage_updates={1:0,2:0,3:0}
        self.global_updates=0
        self.sampler=sampler_factory()
        self._new_optimizer_scaler()
        set_stage(model,connector,1)

    def _new_optimizer_scaler(self):
        self.optimizer=torch.optim.AdamW([
            {'params':list(self.connector.parameters()),'lr':0.0,'name':'connector'},
            {'params':[p for n,p in self.model.named_parameters() if is_lora(n)],'lr':0.0,'name':'lora'}],
            betas=(0.9,0.999),eps=1e-8,weight_decay=0.01,foreach=False,fused=False)
        self.scaler=torch.amp.GradScaler(self.device.type,init_scale=self.initial_scale)

    def snapshot(self):
        if any(p.grad is not None for p in trainable_parameters(self.model,self.connector)):
            raise ValueError('Checkpoints are only valid at cleared complete-update boundaries.')
        return cpu_copy({'schema':'atml-pa3-task2-training-state-v1','identity':self.identity,
            'stage':self.stage,'stage_updates':self.stage_updates,'global_updates':self.global_updates,
            'connector':self.connector.state_dict(),
            'lora':{n:p for n,p in self.model.named_parameters() if is_lora(n)},
            'optimizer':self.optimizer.state_dict(),'scaler':self.scaler.state_dict(),
            'sampler':self.sampler.state_dict(),'rng':capture_rng(self.device)})

    def restore(self,state):
        if state['schema']!='atml-pa3-task2-training-state-v1':raise ValueError('Unknown checkpoint schema.')
        assert_equal(self.identity,state['identity'],'identity')
        self.optimizer.zero_grad(set_to_none=True)
        expected=self.connector.state_dict()
        if expected.keys()!=state['connector'].keys():raise ValueError('Connector checkpoint keys differ.')
        for key,value in expected.items():
            if value.dtype!=state['connector'][key].dtype or value.shape!=state['connector'][key].shape:
                raise ValueError('Connector checkpoint shape or dtype differs.')
        self.connector.load_state_dict(state['connector'],strict=True)
        adapters={n:p for n,p in self.model.named_parameters() if is_lora(n)}
        if adapters.keys()!=state['lora'].keys():raise ValueError('Adapter checkpoint keys differ.')
        with torch.no_grad():
            for name,p in adapters.items():
                value=state['lora'][name]
                if value.dtype!=p.dtype or value.shape!=p.shape:raise ValueError('Adapter shape or dtype differs.')
                p.copy_(value)
        self.stage=state['stage']
        self.stage_updates=copy.deepcopy(state['stage_updates'])
        self.global_updates=state['global_updates']
        set_stage(self.model,self.connector,self.stage)
        assert_precision(self.model,self.connector,self.stage)
        self.sampler=self.sampler_factory()
        self.sampler.load_state_dict(state['sampler'])
        self._new_optimizer_scaler()
        self.optimizer.load_state_dict(state['optimizer'])
        self.scaler.load_state_dict(state['scaler'])
        restore_rng(state['rng'],self.device)

    def select_stage(self,stage):
        # Switching stages retains optimizer moments, scaler, sampler and RNG.
        self.stage=stage
        set_stage(self.model,self.connector,stage)
        return assert_precision(self.model,self.connector,stage)

    def update(self):
        stage=self.stage
        assert_precision(self.model,self.connector,stage)
        u=self.stage_updates[stage]+1
        config=STAGES[stage]
        rates=[learning_rate(config['connector_lr'],stage,u),
            0.0 if config['lora_lr'] is None else learning_rate(config['lora_lr'],stage,u)]
        for group,rate in zip(self.optimizer.param_groups,rates):group['lr']=rate
        pairs=[self.sampler.draw(stage) for _ in range(8)]
        examples=[example for pair in pairs for example in pair['examples']]
        if len(examples)!=16 or any(row['source_split']!='train2014' for row in examples):
            raise ValueError('Training updates require sixteen train-only examples in complete pairs.')
        self.optimizer.zero_grad(set_to_none=True)
        losses=[]
        for start in (0,8):
            rows=examples[start:start+8]
            records=[self.tables['by_text'][(row['user_text'],row['target'])] for row in rows]
            features=self.get_features([(row['source_split'],row['image_id']) for row in rows])
            with torch.autocast(self.device.type,dtype=torch.float16):
                loss=training_loss(self.model,self.connector,records,features)
            losses.append(loss.detach().float().item())
            self.scaler.scale(loss/2).backward()
            del loss,features
        self.scaler.unscale_(self.optimizer)
        connector_nonzero=lora_b_nonzero=lora_a_nonzero=False
        for name,p in [('connector.'+n,p) for n,p in self.connector.named_parameters()]+list(self.model.named_parameters()):
            if not p.requires_grad:
                if p.grad is not None:raise ValueError('Frozen parameter received a gradient: '+name)
                continue
            if p.grad is None or p.grad.dtype!=torch.float32 or not torch.isfinite(p.grad).all():
                raise FloatingPointError('Missing, non-FP32 or nonfinite gradient: '+name)
            nonzero=bool(torch.count_nonzero(p.grad).item())
            if name.startswith('connector.'):connector_nonzero|=nonzero
            if '.lora_B.' in name:lora_b_nonzero|=nonzero
            if '.lora_A.' in name:lora_a_nonzero|=nonzero
        if not connector_nonzero or (stage>=2 and not lora_b_nonzero):
            raise ValueError('Expected connector/active adapter gradient path is absent.')
        norm=torch.nn.utils.clip_grad_norm_(trainable_parameters(self.model,self.connector),1.0,error_if_nonfinite=True)
        scale=self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.scaler.get_scale()<scale:raise FloatingPointError('GradScaler skipped an optimizer step. Stop without advancing counters.')
        self.optimizer.zero_grad(set_to_none=True)
        for state in self.optimizer.state.values():
            for key in ('exp_avg','exp_avg_sq'):
                if state[key].dtype!=torch.float32 or not torch.isfinite(state[key]).all():
                    raise FloatingPointError('Invalid FP32 optimizer moment.')
        if any(not torch.isfinite(p).all() for p in trainable_parameters(self.model,self.connector)):
            raise FloatingPointError('Nonfinite updated parameter.')
        self.global_updates+=1
        self.stage_updates[stage]+=1
        return {'stage':stage,'stage_update':u,'micro_losses':losses,'loss':sum(losses)/2,
            'unclipped_gradient_norm':float(norm),'scale_before':scale,'scale_after':self.scaler.get_scale(),
            'learning_rates':rates,'connector_gradient_nonzero':connector_nonzero,
            'lora_B_gradient_nonzero':lora_b_nonzero,'lora_A_gradient_nonzero':lora_a_nonzero,
            'examples':examples,'pairs':pairs}
