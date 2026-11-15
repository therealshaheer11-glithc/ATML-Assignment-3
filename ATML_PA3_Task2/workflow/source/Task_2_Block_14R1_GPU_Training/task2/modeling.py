"""FP32 connector, standard LoRA stage control and target-only Qwen logits."""
import torch
from torch import nn
from .text import splice_batch,example_mean_ce,IGNORE_INDEX


class Connector(nn.Module):
    def __init__(self,vision_width=768,language_width=1024):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(vision_width,language_width),nn.GELU(),
            nn.Linear(language_width,language_width))
        self.register_buffer('scale',torch.tensor(1.0,dtype=torch.float32))

    def raw(self,features):
        with torch.autocast(features.device.type,enabled=False):
            return self.layers(features.float())

    def forward(self,features,output_dtype=torch.float16):
        with torch.autocast(features.device.type,enabled=False):
            projected=self.raw(features)*self.scale
        return projected.to(output_dtype)

    def set_scale(self,value):
        value=torch.as_tensor(value,dtype=torch.float32,device=self.scale.device)
        if value.numel()!=1 or not torch.isfinite(value).all() or value.item()<=0:
            raise ValueError('Calibration scale must be one positive finite FP32 scalar.')
        self.scale.copy_(value.reshape(()))


def is_lora(name):
    return '.lora_A.' in name or '.lora_B.' in name


def add_standard_lora(base):
    from peft import LoraConfig,TaskType,get_peft_model
    config=LoraConfig(task_type=TaskType.CAUSAL_LM,r=16,lora_alpha=32,lora_dropout=0.05,
        target_modules=['q_proj','k_proj','v_proj','o_proj'],bias='none',init_lora_weights=True)
    model=get_peft_model(base,config,autocast_adapter_dtype=True)
    adapters={name:p for name,p in model.named_parameters() if is_lora(name)}
    if not adapters or any(p.dtype!=torch.float32 for p in adapters.values()):
        raise ValueError('All LoRA parameters must be FP32.')
    if any(torch.count_nonzero(p).item()!=0 for name,p in adapters.items() if '.lora_B.' in name):
        raise ValueError('Initial LoRA B weights must be zero.')
    return model


def set_stage(model,connector,stage,training=True):
    if stage not in (1,2,3):
        raise ValueError('Unknown curriculum stage.')
    if stage==1:
        model.base_model.disable_adapter_layers()
    else:
        model.base_model.enable_adapter_layers()
    for name,parameter in model.named_parameters():
        parameter.requires_grad_(stage>=2 and is_lora(name))
    connector.requires_grad_(True)
    model.train(training)
    connector.train(training)


def assert_precision(model,connector,stage):
    counts={'frozen_base':0,'lora':0,'connector':0,'trainable':0}
    for name,p in model.named_parameters():
        expected=torch.float32 if is_lora(name) else torch.float16
        if p.dtype!=expected:
            raise TypeError('Wrong parameter dtype: '+name)
        if p.requires_grad!=(stage>=2 and is_lora(name)):
            raise ValueError('Wrong stage parameter mask: '+name)
        counts['lora' if is_lora(name) else 'frozen_base']+=p.numel()
        if p.requires_grad:counts['trainable']+=p.numel()
    for p in connector.parameters():
        if p.dtype!=torch.float32 or not p.requires_grad:
            raise TypeError('Connector parameters must remain trainable FP32.')
        counts['connector']+=p.numel()
        counts['trainable']+=p.numel()
    if connector.scale.dtype!=torch.float32 or connector.scale.requires_grad:
        raise TypeError('The fixed calibration scalar must remain non-trainable FP32.')
    return counts


def trainable_parameters(model,connector):
    return list(connector.parameters())+[p for name,p in model.named_parameters() if is_lora(name)]


def target_logits(model,connector,records,features,content_only=False):
    base=model.get_base_model()
    visual=connector(features,base.get_input_embeddings().weight.dtype) if features is not None else None
    batch=splice_batch(records,base.get_input_embeddings(),visual)
    hidden=base.model(inputs_embeds=batch['inputs_embeds'],attention_mask=batch['attention_mask'],
        position_ids=batch['position_ids'],use_cache=False,return_dict=True).last_hidden_state
    mask=batch['labels'][:,1:]!=IGNORE_INDEX
    if content_only:mask=mask & batch['content_mask'][:,1:]
    selected=hidden[:,:-1][mask]
    logits=base.lm_head(selected)
    targets=batch['labels'][:,1:][mask]
    ids=torch.arange(len(records),device=targets.device)[:,None].expand_as(mask)[mask]
    return logits,targets,ids,batch


def training_loss(model,connector,records,features):
    logits,targets,ids,batch=target_logits(model,connector,records,features)
    return example_mean_ce(logits,targets,ids,len(records))


def calibrate(model,connector,calibration,get_features,device):
    """Global FP32 streaming RMS, preserving the saved sample order."""
    embedding=model.get_base_model().get_input_embeddings()
    text_sum=torch.zeros((),dtype=torch.float32,device=device)
    image_sum=torch.zeros((),dtype=torch.float32,device=device)
    text_count=image_count=0
    with torch.no_grad(),torch.autocast(torch.device(device).type,enabled=False):
        entries=calibration['caption_content']
        for start in range(0,len(entries),32):
            ids=[token for row in entries[start:start+32] for token in row['content_ids']]
            values=embedding(torch.tensor(ids,device=device,dtype=torch.long)).float()
            text_sum+=values.square().sum(dtype=torch.float32)
            text_count+=values.numel()
        images=calibration['image_ids']
        for start in range(0,len(images),8):
            features=get_features([('train2014',image_id) for image_id in images[start:start+8]])
            values=connector.raw(features)
            image_sum+=values.square().sum(dtype=torch.float32)
            image_count+=values.numel()
        text_rms=torch.sqrt(text_sum/text_count)
        image_rms=torch.sqrt(image_sum/image_count)
        if not torch.isfinite(text_rms).item() or not torch.isfinite(image_rms).item() or min(text_rms.item(),image_rms.item())<=0:
            raise FloatingPointError('Zero or nonfinite calibration RMS. No clamp or epsilon was applied.')
        scale=text_rms/image_rms
        connector.set_scale(scale)
    return {'text_sum_squares_fp32':text_sum.item(),'visual_sum_squares_fp32':image_sum.item(),
        'text_elements':text_count,'visual_elements':image_count,'text_rms':text_rms.item(),
        'visual_rms':image_rms.item(),'scale_fp32':scale.item(),'image_ids':calibration['image_ids'],
        'caption_ids':calibration['caption_ids'],'text_caption_chunk':32,'image_chunk':8,
        'arithmetic':'FP32 streaming global sums, RMS and ratio','evaluation_inputs_used':False}
