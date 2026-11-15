"""Native Qwen prompt layouts, verbatim targets and continuous-vision batch masks."""
import hashlib
import json

CAPTION_USER_TEXT='Describe this image.'
IGNORE_INDEX=-100


def content_ids(tokenizer,text,maximum,field):
    if not isinstance(text,str) or not text.strip():
        raise ValueError(field+' must contain nonempty original text.')
    ids=list(tokenizer.encode(text,add_special_tokens=False))
    if not ids or len(ids)>maximum:
        raise ValueError(f'{field} has {len(ids)} tokens and exceeds its approved cap {maximum}. No truncation was applied.')
    reserved={tokenizer.eos_token_id,tokenizer.pad_token_id,tokenizer.convert_tokens_to_ids('<|im_start|>')}
    if any(token in reserved for token in ids):
        raise ValueError(field+' contains a reserved chat control token.')
    if tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)!=text:
        raise ValueError(field+' does not round-trip exactly through the pinned tokenizer.')
    return ids


def native_prompt(tokenizer,user_text):
    content_ids(tokenizer,user_text,80,'Complete user text')
    messages=[{'role':'user','content':user_text}]
    rendered=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    if rendered.count(user_text)!=1:
        raise ValueError('Cannot identify the unchanged user message in the native template.')
    prefix,suffix=rendered.split(user_text,1)
    if prefix!='<|im_start|>user\n' or '<|im_start|>system\n' in rendered:
        raise ValueError('Unexpected native user-message header or added system prompt.')
    prefix_ids=list(tokenizer.encode(prefix,add_special_tokens=False))
    prompt_ids=list(tokenizer.encode(rendered,add_special_tokens=False))
    native_ids=list(tokenizer.apply_chat_template(messages,tokenize=True,return_dict=False,
        add_generation_prompt=True,enable_thinking=False))
    if prompt_ids!=native_ids or prompt_ids[:len(prefix_ids)]!=prefix_ids:
        raise ValueError('Native tokenization does not preserve the visual insertion boundary.')
    if tokenizer.decode(prompt_ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)!=rendered:
        raise ValueError('Native prompt tokenization changed its text.')
    return {'prompt_ids':prompt_ids,'user_prefix_length':len(prefix_ids),
        'native_prompt_text':rendered,'native_assistant_suffix':suffix}


def prepare_text(tokenizer,user_text,target,target_limit,question=None):
    if tokenizer.eos_token_id!=151645 or tokenizer.pad_token_id!=151643:
        raise ValueError('Qwen special token identities differ from the approved tokenizer.')
    if question is not None:
        content_ids(tokenizer,question,80,'Original question')
        if user_text!='Answer with a short phrase.\n'+question:
            raise ValueError('The original VQA question or approved user text changed.')
    prompt=native_prompt(tokenizer,user_text)
    target_ids=content_ids(tokenizer,target,target_limit,'Original target')
    # Continuation tokenization preserves the raw original caption, including
    # leading/trailing newlines. Full-message serialization can trim newlines.
    completion_ids=target_ids+[tokenizer.eos_token_id]
    decoded=tokenizer.decode(prompt['prompt_ids']+completion_ids,skip_special_tokens=False,
        clean_up_tokenization_spaces=False)
    expected=prompt['native_prompt_text']+target+tokenizer.eos_token
    if decoded!=expected:
        raise ValueError('The supervised completion does not preserve its original target verbatim.')
    return {'user_text':user_text,'target':target,'prompt_ids':prompt['prompt_ids'],
        'user_prefix_length':prompt['user_prefix_length'],'target_ids':target_ids,
        'end_token_id':tokenizer.eos_token_id,'target_content_tokens':len(target_ids)}


def splice_batch(records,token_embedding,visual_embeddings=None,pad_token_id=151643,include_targets=True,padding_side=None):
    """Insert continuous patches after the native user header and mask all context."""
    import torch
    if not records:
        raise ValueError('A nonempty batch is required.')
    if token_embedding.weight.requires_grad:
        raise ValueError('The Qwen token embedding must remain frozen.')
    padding_side=padding_side or ('right' if include_targets else 'left')
    if padding_side not in ('right','left'):
        raise ValueError('Unsupported padding side.')
    device,dtype=token_embedding.weight.device,token_embedding.weight.dtype
    width=token_embedding.weight.shape[1]
    if visual_embeddings is not None:
        if tuple(visual_embeddings.shape)!=(len(records),196,width):
            raise ValueError('Continuous visual embeddings must have batch x 196 x language width shape.')
        visual_embeddings=visual_embeddings.to(device=device,dtype=dtype)
    rows,label_rows,content_rows=[],[],[]
    target_starts=[]
    for index,record in enumerate(records):
        prompt,target=record['prompt_ids'],record['target_ids']
        prefix=record['user_prefix_length']
        if not 0<prefix<len(prompt) or not target or record['end_token_id']!=151645:
            raise ValueError('Invalid prepared text record.')
        text_ids=prompt+(target+[record['end_token_id']] if include_targets else [])
        text=token_embedding(torch.tensor(text_ids,device=device,dtype=torch.long))
        pieces=[text[:prefix]]
        visual_length=0
        if visual_embeddings is not None:
            pieces.append(visual_embeddings[index])
            visual_length=196
        pieces.append(text[prefix:])
        row=torch.cat(pieces,dim=0)
        labels=torch.full((len(row),),IGNORE_INDEX,device=device,dtype=torch.long)
        content_mask=torch.zeros((len(row),),device=device,dtype=torch.bool)
        start=len(prompt)+visual_length
        if include_targets:
            labels[start:]=torch.tensor(target+[record['end_token_id']],device=device,dtype=torch.long)
            content_mask[start:start+len(target)]=True
        rows.append(row)
        label_rows.append(labels)
        content_rows.append(content_mask)
        target_starts.append(start)
    lengths=[len(row) for row in rows]
    maximum=max(lengths)
    pad=token_embedding(torch.tensor([pad_token_id],device=device,dtype=torch.long))
    padded=[]
    attention=torch.zeros((len(rows),maximum),device=device,dtype=torch.long)
    labels=torch.full((len(rows),maximum),IGNORE_INDEX,device=device,dtype=torch.long)
    content=torch.zeros((len(rows),maximum),device=device,dtype=torch.bool)
    for index,(row,length) in enumerate(zip(rows,lengths)):
        offset=0 if padding_side=='right' else maximum-length
        padding=pad.expand(maximum-length,-1)
        padded.append(torch.cat([row,padding] if padding_side=='right' else [padding,row],dim=0))
        attention[index,offset:offset+length]=1
        labels[index,offset:offset+length]=label_rows[index]
        content[index,offset:offset+length]=content_rows[index]
        target_starts[index]+=offset
    positions=attention.cumsum(-1)-1
    positions.masked_fill_(attention==0,0)
    return {'inputs_embeds':torch.stack(padded),'attention_mask':attention,'position_ids':positions,
        'labels':labels,'content_mask':content,'lengths':lengths,'target_starts':target_starts,'padding_side':padding_side}


def example_mean_ce(logits,targets,example_ids,batch_size):
    """Mean token CE within each example, then equal-weight mean across examples."""
    import torch
    import torch.nn.functional as F
    if logits.ndim!=2 or targets.ndim!=1 or example_ids.shape!=targets.shape or logits.shape[0]!=len(targets):
        raise ValueError('Selected logits, targets and example identities do not align.')
    if type(batch_size) is not int or batch_size<=0 or (example_ids<0).any() or (example_ids>=batch_size).any():
        raise ValueError('Invalid example identities for the requested batch size.')
    losses=F.cross_entropy(logits.float(),targets,reduction='none')
    means=[]
    for index in range(batch_size):
        selected=losses[example_ids==index]
        if selected.numel()==0:
            raise ValueError('An example has no supervised target tokens.')
        means.append(selected.mean())
    result=torch.stack(means).mean()
    if not torch.isfinite(result).item():
        raise FloatingPointError('Nonfinite target loss.')
    return result


def causal_example_loss(logits,labels,content_mask=None):
    import torch
    mask=labels[:,1:]!=IGNORE_INDEX
    if content_mask is not None:
        mask=mask & content_mask[:,1:]
    example_ids=torch.arange(labels.shape[0],device=labels.device)[:,None].expand_as(mask)[mask]
    return example_mean_ce(logits[:,:-1][mask],labels[:,1:][mask],example_ids,labels.shape[0])
